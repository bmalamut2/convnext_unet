import os, sys, math, gc, time, glob, random, warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict
warnings.filterwarnings("ignore")

BASE_PATH = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
import cv2
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

class CFG:
    DATA_ROOT = Path(os.environ.get("DATA_ROOT", BASE_PATH / "pneumothorax"))  # default dataset bundle
    WORK_DIR  = Path(os.environ.get("WORK_DIR", BASE_PATH / "outputs"))        # store outputs next to script
    CSV_CANDIDATES = ["train-rle.csv", "stage_2_train.csv"]  # common names
    IMG_EXTS = (".png", ".jpg", ".jpeg")
    DCM_EXT  = ".dcm"

    HEIGHT = 1024
    WIDTH  = 1024
    RESIZE = 1024            # final training size (square)

    FOLDS = 5
    SEED = 42
    FOLD_TO_RUN = 0          # change to train a different fold

    # training
    EPOCHS = 20
    TRAIN_BS = 4             # adjust per GPU memory
    VALID_BS = 4
    ACCUM_STEPS = 1
    NUM_WORKERS = 2
    AMP = True

    # optimizer / schedule
    LR = 3e-4
    WD = 1e-4
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # losses
    DICE_WEIGHT = 0.5
    BCE_WEIGHT  = 0.5
    CLS_WEIGHT  = 0.2         # aux classification head loss weight

    # post-processing (triplet threshold)
    PRESENT_THR = 0.70        # threshold to decide "any pneumothorax present?"
    MIN_AREA    = 600         # minimal area at PRESENT_THR to call positive
    BIN_THR     = 0.30        # binarization threshold for final mask

    # model
    # SMP uses timm encoders; include several fallback families for older pkg versions.
    ENCODER_CANDIDATES = [
        "tu-convnext_base",
        "timm-convnext_base",
        "convnext_base",
        "mit_b4",
        "mit_b3",
        "efficientnet-b4",
        "resnet34",
    ]
    ENCODER_WEIGHTS = "imagenet"   # ImageNet pretraining
    IN_CHANNELS = 1
    CLASSES = 1
    DECODER_ATTENTION = "scse"     # fallback if not supported
    DROPOUT = 0.3                  # for aux classification head

    # split
    VALID_SPLIT_SEED = 13  # seed for fold split

# reproducibility
def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CFG.WORK_DIR.mkdir(parents=True, exist_ok=True)
seed_everything(CFG.SEED)

def find_csv(root: Path) -> Path:
    for name in CFG.CSV_CANDIDATES:
        p = root / name
        if p.exists():
            return p
    # look for any csv with EncodedPixels
    for p in root.glob("*.csv"):
        try:
            df = pd.read_csv(p, nrows=2)
            if "EncodedPixels" in df.columns:
                return p
        except Exception:
            pass
    raise FileNotFoundError("Could not find an RLE CSV (e.g., train-rle.csv or stage_2_train.csv).")

def build_image_index(root: Path) -> Dict[str, Path]:
    """Index all candidate image files (PNG/JPG/DCM) by basename (ImageId)."""
    mapping = {}
    # 1) images in common folders
    candidates = []
    for sub in ["", "train", "train_png", "train_images", "images", "dicom-images-train"]:
        p = root / sub
        if p.exists():
            candidates.append(p)
    if not candidates:
        candidates = [root]

    # find PNG/JPG
    for base in candidates:
        for ext in CFG.IMG_EXTS:
            for fp in base.rglob(f"*{ext}"):
                key = fp.stem  # ImageId without extension
                if key not in mapping:
                    mapping[key] = fp

    # find DICOM
    for base in candidates:
        for fp in base.rglob(f"*{CFG.DCM_EXT}"):
            key = fp.stem
            if key not in mapping:
                mapping[key] = fp

    if len(mapping) == 0:
        raise FileNotFoundError("No images found (PNG/JPG or DICOM). Check DATA_ROOT.")
    return mapping

def normalize_rle_encoding(enc) -> Optional[str]:
    if enc is None:
        return None
    if isinstance(enc, float) and np.isnan(enc):
        return None
    s = str(enc).strip()
    if s == "" or s == "-1":
        return None
    return s

def normalize_rle_list(enc_list: List[str]) -> List[str]:
    cleaned: List[str] = []
    for enc in enc_list:
        norm = normalize_rle_encoding(enc)
        if norm is not None:
            cleaned.append(norm)
    return cleaned

def rle_decode(rle_str: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode RLE as (height, width) binary mask. RLE is 'start length start length ...'"""
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle_str, float):
        if np.isnan(rle_str):
            return mask.reshape(h, w)
        rle_str = str(rle_str)
    rle_str = str(rle_str).strip()
    if rle_str == "-1" or rle_str == "":
        return mask.reshape(h, w)
    s = list(map(int, rle_str.strip().split()))
    starts, lengths = s[0::2], s[1::2]
    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start+length] = 1
    return mask.reshape(w, h).T  # Kaggle format is column-major

def build_masks_for_id(enc_list: List[str], shape: Tuple[int, int]) -> np.ndarray:
    """Combine multiple RLE rows for one image id into one mask."""
    enc_list = normalize_rle_list(enc_list)
    if len(enc_list) == 0:
        return np.zeros(shape, np.uint8)
    out = np.zeros(shape, np.uint8)
    for enc in enc_list:
        out |= rle_decode(str(enc), shape)
    return out

def load_png(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image {path}")
    return img

def load_dicom_image(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    # apply rescale if present
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    img = img * slope + intercept
    # simple robust normalization
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    img = (img * 255.0).astype(np.uint8)
    return img

class PneumoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id2path: Dict[str, Path], transforms=None, image_size=1024):
        self.df = df.reset_index(drop=True)
        self.id2path = id2path
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["ImageId"]
        enc_list = row["Encodings"]  # list of strings (may be empty)

        fp = self.id2path[image_id]
        if fp.suffix.lower() in CFG.IMG_EXTS:
            img = load_png(fp)
        else:
            img = load_dicom_image(fp)

        # ensure HxW -> resize to target square
        img = cv2.resize(img, (CFG.RESIZE, CFG.RESIZE), interpolation=cv2.INTER_AREA)

        # build mask
        mask = build_masks_for_id(enc_list, (CFG.HEIGHT, CFG.WIDTH))
        mask = cv2.resize(mask, (CFG.RESIZE, CFG.RESIZE), interpolation=cv2.INTER_NEAREST)

        # albumentations expects HWC
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        img = np.expand_dims(img, -1)  # (H,W,1)

        if self.transforms:
            data = self.transforms(image=img, mask=mask)
            img, mask = data["image"], data["mask"]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask[None]).float()

        # binary presence label for aux cls head
        has_mask = 1.0 if mask.sum() > 0 else 0.0
        return img, mask, torch.tensor([has_mask], dtype=torch.float32), image_id

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=7, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ElasticTransform(alpha=20, sigma=4, alpha_affine=0, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.CoarseDropout(max_holes=2, max_height=int(0.06*CFG.RESIZE), max_width=int(0.06*CFG.RESIZE), fill_value=0, p=0.2),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(transpose_mask=True),
    ])

def get_valid_transforms():
    return A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(transpose_mask=True),
    ])

def _has_invalid_conv(model: nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.out_channels <= 0:
            return True
    return False

def build_model() -> nn.Module:
    aux_params = dict(pooling="avg", dropout=CFG.DROPOUT, classes=1, activation=None)
    last_ex = None
    attention_candidates = [CFG.DECODER_ATTENTION, None] if CFG.DECODER_ATTENTION else [None]
    for name in CFG.ENCODER_CANDIDATES:
        for attention in attention_candidates:
            kwargs = dict(
                encoder_name=name,
                encoder_weights=CFG.ENCODER_WEIGHTS,
                in_channels=CFG.IN_CHANNELS,
                classes=CFG.CLASSES,
                aux_params=aux_params,
            )
            if attention:
                kwargs["decoder_attention_type"] = attention
            try:
                model = smp.UnetPlusPlus(**kwargs)
                att_label = attention if attention else "none"
                if _has_invalid_conv(model):
                    print(f"[Model] encoder={name} attention={att_label} produced zero-channel convs; trying next option.")
                    continue
                print(f"[Model] Using encoder_name={name}, attention={att_label}")
                return model
            except Exception as ex:
                last_ex = ex
                continue
    raise RuntimeError(f"Could not construct model with ConvNeXt encoders. Last error: {last_ex}")

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        # logits: (N,1,H,W) raw; targets: (N,1,H,W) {0,1}
        probs = torch.sigmoid(logits)
        dims = (0,2,3)
        num = 2.0*(probs*targets).sum(dims) + self.smooth
        den = probs.sum(dims) + targets.sum(dims) + self.smooth
        loss = 1.0 - (num/den)
        return loss.mean()

bce_criterion  = nn.BCEWithLogitsLoss()
dice_criterion = DiceLoss()

def bce_dice_loss(logits, targets):
    return CFG.BCE_WEIGHT*bce_criterion(logits, targets) + CFG.DICE_WEIGHT*dice_criterion(logits, targets)

@torch.no_grad()
def compute_dice_per_image(pred_mask_bin: np.ndarray, gt_mask: np.ndarray) -> float:
    # both shapes (H,W) {0,1}
    p_sum = pred_mask_bin.sum()
    g_sum = gt_mask.sum()
    if p_sum == 0 and g_sum == 0:
        return 1.0
    inter = (pred_mask_bin & gt_mask).sum()
    return (2.0*inter) / (p_sum + g_sum + 1e-6)

def postprocess_triplet(prob: np.ndarray, present_thr=0.7, min_area=600, bin_thr=0.3) -> np.ndarray:
    """prob: (H,W) in [0,1]; returns binary mask (H,W)."""
    # presence decision at present_thr with min area
    bin_present = (prob >= present_thr).astype(np.uint8)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_present, connectivity=8)
    present = False
    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            present = True
            break
    if not present:
        return np.zeros_like(bin_present, dtype=np.uint8)
    # final mask at bin_thr
    return (prob >= bin_thr).astype(np.uint8)

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    running = 0.0
    for step, (imgs, masks, cls_targets, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        cls_targets = cls_targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=CFG.AMP):
            out = model(imgs)
            if isinstance(out, (list, tuple)) and len(out) == 2:
                seg_logits, cls_logits = out
            else:
                seg_logits, cls_logits = out, None

            loss_seg = bce_dice_loss(seg_logits, masks)
            loss = loss_seg
            if cls_logits is not None:
                # BCEWithLogits for presence
                loss_cls = bce_criterion(cls_logits.view(-1), cls_targets.view(-1))
                loss = loss + CFG.CLS_WEIGHT*loss_cls

        optimizer.zero_grad(set_to_none=True)
        if CFG.AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running += loss.item()
    return running / max(1, len(loader))

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    dices = []
    for imgs, masks, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out = model(imgs)
        seg_logits = out[0] if (isinstance(out, (list, tuple)) and len(out) == 2) else out
        probs = torch.sigmoid(seg_logits).float().cpu().numpy()

        for i in range(probs.shape[0]):
            p = probs[i,0]
            g = masks[i,0].cpu().numpy().astype(np.uint8)
            # apply triplet post-proc for scoring
            p_bin = postprocess_triplet(p, CFG.PRESENT_THR, CFG.MIN_AREA, CFG.BIN_THR)
            dice = compute_dice_per_image(p_bin, g)
            dices.append(dice)
    return float(np.mean(dices))

def build_dataframe(rle_csv: Path, id2path: Dict[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(rle_csv)
    # standardize column names
    if "ImageId" not in df.columns:
        # some CSVs use "ImageId" or "ImageId_dicom"
        for c in df.columns:
            if "ImageId" in c:
                df = df.rename(columns={c: "ImageId"})
                break
    if "EncodedPixels" not in df.columns:
        # sometimes it's " EncodedPixels"
        for c in df.columns:
            if "EncodedPixels" in c:
                df = df.rename(columns={c: "EncodedPixels"})
                break

    # group rows for same image
    grouped = (df.groupby("ImageId")["EncodedPixels"]
                 .apply(list)
                 .reset_index()
                 .rename(columns={"EncodedPixels":"Encodings"}))

    # keep only those we can find on disk
    grouped = grouped[grouped["ImageId"].isin(id2path.keys())].copy()
    grouped["Encodings"] = grouped["Encodings"].apply(normalize_rle_list)
    grouped["HasMask"] = grouped["Encodings"].apply(lambda enc_list: int(len(enc_list) > 0))
    return grouped

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rle_csv = find_csv(CFG.DATA_ROOT)
    print(f"Using RLE CSV: {rle_csv.name}")

    id2path = build_image_index(CFG.DATA_ROOT)
    print(f"Indexed {len(id2path)} images (PNG/JPG/DCM).")

    df = build_dataframe(rle_csv, id2path)
    print(f"DataFrame rows (unique images): {len(df)}, positive ratio: {df['HasMask'].mean():.3f}")

    # Stratified K-Fold by HasMask
    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.VALID_SPLIT_SEED)
    splits = list(skf.split(df, df["HasMask"]))

    tr_idx, va_idx = splits[CFG.FOLD_TO_RUN]
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)
    print(f"Fold {CFG.FOLD_TO_RUN}: train={len(df_tr)}, valid={len(df_va)}")

    train_ds = PneumoDataset(df_tr, id2path, transforms=get_train_transforms(), image_size=CFG.RESIZE)
    valid_ds = PneumoDataset(df_va, id2path, transforms=get_valid_transforms(), image_size=CFG.RESIZE)

    train_loader = DataLoader(train_ds, batch_size=CFG.TRAIN_BS, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=CFG.VALID_BS, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model = build_model().to(device)
    optimizer = AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_MAX, eta_min=CFG.MIN_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.AMP)

    best_dice = -1.0
    best_path = CFG.WORK_DIR / f"unetpp_convnext_fold{CFG.FOLD_TO_RUN}_best.pt"

    for epoch in range(1, CFG.EPOCHS+1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_dice = validate_one_epoch(model, valid_loader, device)
        scheduler.step()

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{CFG.EPOCHS} | train_loss={tr_loss:.4f} | val_dice={val_dice:.4f} | time={dt:.1f}s")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(),
                        "cfg": CFG.__dict__,
                        "val_dice": best_dice}, best_path)
            print(f"  -> saved new best to {best_path} (dice={best_dice:.4f})")

    print(f"Training done. Best val Dice={best_dice:.4f}. Weights: {best_path}")

if __name__ == "__main__":
    main()
