 
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

from monai.networks.nets import UNet

 
K = 2
ROI_MARGIN = 10
THRESHOLD = 0.6
AGGREGATION = "max"
ROI_SIZE = 224
BATCH_SIZE = 4
SEED = 42
SEG_THRESHOLD = 0.5                
VALID_AREA_RATIO_THRESHOLD = 0.03 
DEVICE_STR = "cuda"          

 
INDEX_CSV = "/path/to/your/slice_index.csv"      
IMAGE_ROOT = "/"                                 
SEG_MODEL_PATH = "/path/to/segmentation_model.pt"
CLS_MODEL_PATH = "/path/to/classifier_model.pt"
OUTPUT_DIR = "/path/to/output_directory"

# ============================================================
# 辅助函数
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_unet(channels, num_res_units=None):
    kwargs = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": channels,
        "strides": (2, 2, 2, 2),
    }
    if num_res_units is not None:
        kwargs["num_res_units"] = num_res_units
    return UNet(**kwargs)

def clean_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    new_ckpt = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("module."):
            nk = nk.replace("module.", "", 1)
        new_ckpt[nk] = v
    return new_ckpt

def load_segmentation_model(weight_path, device):
    ckpt = torch.load(weight_path, map_location=device)
    ckpt = clean_state_dict(ckpt)
    channel_candidates = [
        (16, 32, 64, 128, 256),
        (32, 64, 128, 256, 512),
        (64, 128, 256, 512, 1024),
    ]
    num_res_candidates = [None, 0, 1, 2]
    last_error = None
    for channels in channel_candidates:
        for num_res_units in num_res_candidates:
            try:
                model = build_unet(channels, num_res_units).to(device)
                model.load_state_dict(ckpt, strict=True)
                model.eval()
                return model, channels, num_res_units
            except Exception as e:
                last_error = e
    raise RuntimeError("Failed to load segmentation model.") from last_error

def build_classifier():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def load_classifier_model(weight_path, device):
    model = build_classifier().to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def resolve_path(path_value, root_dir):
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return Path(root_dir) / path

def load_gray_float(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1:
        arr = arr / 255.0
    return np.clip(arr, 0, 1).astype(np.float32)

def load_gray_uint8(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def pad_to_multiple(x, multiple=16):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return x, h, w

@torch.no_grad()
def predict_segmentation_probability(model, image_path, device):
    arr = load_gray_float(image_path)
    h, w = arr.shape
    x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(device)
    x, original_h, original_w = pad_to_multiple(x, multiple=16)
    logits = model(x)
    prob = torch.sigmoid(logits)
    prob = prob[:, :, :original_h, :original_w]
    return prob.squeeze().detach().cpu().numpy().astype(np.float32)

def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def expand_bbox(bbox, h, w, margin):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w - 1, x2 + margin)
    y2 = min(h - 1, y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def get_connected_components(mask):
    labeled, num = ndimage.label(mask)
    if num == 0:
        return labeled, num, np.array([])
    comp_sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    return labeled, num, np.asarray(comp_sizes, dtype=np.float32)

def generate_probability_maps(index_df, seg_model, output_dir, device):
    prob_dir = output_dir / "probability_maps"
    prob_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="Generating probability maps"):
        pid = str(row["pid"])
        sid = str(row["slice_id"]).zfill(3)
        image_path = Path(row["image_path_resolved"])
        if not image_path.exists():
            continue
        prob_path = prob_dir / f"{pid}_slice_{sid}.npy"
        if not prob_path.exists():
            prob = predict_segmentation_probability(seg_model, image_path, device)
            np.save(prob_path, prob)
        new_row = row.to_dict()
        new_row["prob_path"] = str(prob_path)
        rows.append(new_row)
    prob_df = pd.DataFrame(rows)
    prob_df.to_csv(output_dir / "probability_map_index.csv", index=False)
    return prob_df

def build_patient_slice_table(pid, patient_df, seg_threshold, valid_area_ratio_threshold):
    rows = []
    patient_df = patient_df.sort_values("slice_id").reset_index(drop=True)
    for _, row in patient_df.iterrows():
        sid = str(row["slice_id"]).zfill(3)
        image_path = Path(row["image_path_resolved"])
        prob_path = Path(row["prob_path"])
        if not image_path.exists() or not prob_path.exists():
            continue
        prob = np.load(prob_path).astype(np.float32)
        h, w = prob.shape
        area_total = h * w
        mask = prob >= seg_threshold
        area = int(mask.sum())
        area_ratio = area / max(area_total, 1)
        if area > 0:
            mean_confidence = float(prob[mask].mean())
            max_confidence = float(prob.max())
            evidence_score = float((prob * mask).sum() / max(area_total, 1))
        else:
            mean_confidence = 0.0
            max_confidence = float(prob.max())
            evidence_score = 0.0
        if area_ratio >= valid_area_ratio_threshold:
            confidence_score = mean_confidence
        else:
            confidence_score = 0.0
        label = int(row["label"]) if "label" in row and not pd.isna(row["label"]) else -1
        center = str(row["center"]) if "center" in row and not pd.isna(row["center"]) else "unknown"
        rows.append({
            "pid": str(pid),
            "center": center,
            "sid": sid,
            "image_path": str(image_path),
            "prob_path": str(prob_path),
            "label": label,
            "area": area,
            "area_ratio": area_ratio,
            "mean_confidence": mean_confidence,
            "max_confidence": max_confidence,
            "evidence_score": evidence_score,
            "confidence_score": confidence_score,
        })
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sid").reset_index(drop=True)

def select_best_contiguous_window(patient_slice_df, k):
    patient_slice_df = patient_slice_df.sort_values("sid").reset_index(drop=True)
    n = len(patient_slice_df)
    if n == 0:
        return None
    if n <= k:
        selected = patient_slice_df.copy()
        selected["window_score"] = selected["confidence_score"].sum()
        selected["window_start_sid"] = selected["sid"].min()
        selected["window_end_sid"] = selected["sid"].max()
        return selected
    windows = []
    for start in range(0, n - k + 1):
        sub = patient_slice_df.iloc[start:start + k].copy()
        windows.append({
            "start": start,
            "end": start + k - 1,
            "start_sid": str(sub.iloc[0]["sid"]),
            "end_sid": str(sub.iloc[-1]["sid"]),
            "window_score": float(sub["confidence_score"].sum()),
            "mean_confidence": float(sub["mean_confidence"].mean()),
            "mean_area_ratio": float(sub["area_ratio"].mean()),
            "mean_evidence": float(sub["evidence_score"].mean()),
        })
    window_df = pd.DataFrame(windows)
    best = window_df.sort_values("window_score", ascending=False).iloc[0]
    selected = patient_slice_df.iloc[int(best["start"]):int(best["end"]) + 1].copy()
    selected["window_score"] = float(best["window_score"])
    selected["window_start_sid"] = str(best["start_sid"])
    selected["window_end_sid"] = str(best["end_sid"])
    return selected

def crop_roi(image_path, prob_path, output_path, seg_threshold, margin, roi_size):
    image = load_gray_uint8(image_path)
    prob = np.load(prob_path).astype(np.float32)
    if image.shape != prob.shape:
        return False
    mask = prob >= seg_threshold
    labeled, num, comp_sizes = get_connected_components(mask)
    if num == 0:
        return False
    max_idx = int(np.argmax(comp_sizes) + 1)
    largest_component = labeled == max_idx
    bbox = bbox_from_mask(largest_component)
    if bbox is None:
        return False
    h, w = image.shape
    bbox = expand_bbox(bbox, h, w, margin)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2+1, x1:x2+1]
    if roi.size == 0:
        return False
    roi_img = Image.fromarray(roi.astype(np.uint8)).convert("L")
    roi_img = roi_img.resize((roi_size, roi_size), Image.BILINEAR)
    roi_img.save(output_path)
    return True

def build_roi_dataset(prob_df, output_dir, k, margin, roi_size, seg_threshold, valid_area_ratio_threshold):
    roi_dir = output_dir / f"roi_K{k}_margin{margin}"
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_rows = []
    window_rows = []
    failed_pids = []
    for pid, patient_df in tqdm(prob_df.groupby("pid"), desc="Building ROI dataset"):
        patient_slice_df = build_patient_slice_table(pid, patient_df, seg_threshold, valid_area_ratio_threshold)
        if len(patient_slice_df) == 0:
            failed_pids.append(pid)
            continue
        selected_df = select_best_contiguous_window(patient_slice_df, k)
        if selected_df is None or len(selected_df) == 0:
            failed_pids.append(pid)
            continue
        label = int(selected_df["label"].iloc[0])
        center = str(selected_df["center"].iloc[0])
        window_rows.append({
            "pid": pid,
            "center": center,
            "label": label,
            "k": k,
            "roi_margin": margin,
            "window_start_sid": selected_df["window_start_sid"].iloc[0],
            "window_end_sid": selected_df["window_end_sid"].iloc[0],
            "window_score": float(selected_df["window_score"].iloc[0]),
            "n_selected": int(len(selected_df)),
            "mean_area_ratio": float(selected_df["area_ratio"].mean()),
            "mean_confidence": float(selected_df["mean_confidence"].mean()),
            "mean_evidence": float(selected_df["evidence_score"].mean()),
        })
        for _, row in selected_df.iterrows():
            sid = str(row["sid"]).zfill(3)
            image_path = Path(row["image_path"])
            prob_path = Path(row["prob_path"])
            roi_path = roi_dir / f"{pid}_slice_{sid}_roi.png"
            if not roi_path.exists():
                ok = crop_roi(image_path, prob_path, roi_path, seg_threshold, margin, roi_size)
            else:
                ok = True
            if ok:
                roi_rows.append({
                    "pid": pid,
                    "center": center,
                    "sid": sid,
                    "roi_path": str(roi_path),
                    "label": label,
                    "k": k,
                    "roi_margin": margin,
                })
    roi_df = pd.DataFrame(roi_rows)
    window_df = pd.DataFrame(window_rows)
    failed_df = pd.DataFrame({"pid": failed_pids})
    roi_df.to_csv(output_dir / f"roi_dataset_K{k}_margin{margin}.csv", index=False)
    window_df.to_csv(output_dir / f"selected_windows_K{k}_margin{margin}.csv", index=False)
    failed_df.to_csv(output_dir / "failed_patients.csv", index=False)
    return roi_df, window_df, failed_df

class ROIDataset(Dataset):
    def __init__(self, dataframe, roi_size):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])
        self.roi_size = roi_size
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["roi_path"]).convert("RGB")
        image = image.resize((self.roi_size, self.roi_size), Image.BILINEAR)
        x = self.transform(image)
        label = int(row["label"]) if "label" in row and not pd.isna(row["label"]) else -1
        return {
            "image": x,
            "pid": str(row["pid"]),
            "center": str(row["center"]),
            "sid": str(row["sid"]),
            "label": torch.tensor(float(label), dtype=torch.float32),
        }

@torch.no_grad()
def predict_slice_probabilities(model, roi_df, batch_size, roi_size, device):
    dataset = ROIDataset(roi_df, roi_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    rows = []
    for batch in tqdm(loader, desc="Running classification inference"):
        x = batch["image"].to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        labels = batch["label"].cpu().numpy()
        pids = batch["pid"]
        centers = batch["center"]
        sids = batch["sid"]
        for pid, center, sid, label, prob in zip(pids, centers, sids, labels, probs):
            rows.append({
                "pid": str(pid),
                "center": str(center),
                "sid": str(sid),
                "label": int(label),
                "prob": float(prob),
            })
    return pd.DataFrame(rows)

def aggregate_patient_probabilities(slice_df, method):
    rows = []
    for pid, patient_df in slice_df.groupby("pid"):
        probs = np.sort(patient_df["prob"].values.astype(float))[::-1]
        label = int(patient_df["label"].iloc[0])
        center = str(patient_df["center"].iloc[0])
        if method == "max":
            patient_prob = float(np.max(probs))
        elif method == "mean":
            patient_prob = float(np.mean(probs))
        elif method == "top2mean":
            patient_prob = float(np.mean(probs[:min(2, len(probs))]))
        elif method == "top3mean":
            patient_prob = float(np.mean(probs[:min(3, len(probs))]))
        else:
            raise ValueError(f"Unsupported aggregation: {method}")
        rows.append({
            "pid": pid,
            "center": center,
            "label": label,
            "prob": patient_prob,
        })
    return pd.DataFrame(rows)

def assign_prediction(patient_df, threshold):
    patient_df = patient_df.copy()
    patient_df["pred_label"] = (patient_df["prob"] >= threshold).astype(int)
    patient_df["prediction"] = patient_df["pred_label"].map({1: "PAS", 0: "non-PAS"})
    return patient_df

def prepare_index(index_csv, image_root):
    df = pd.read_csv(index_csv)
    required = ["pid", "slice_id", "image_path"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["pid"] = df["pid"].astype(str)
    df["slice_id"] = df["slice_id"].astype(str).str.zfill(3)
    df["image_path_resolved"] = df["image_path"].apply(lambda x: str(resolve_path(x, image_root)))
    if "label" not in df.columns:
        df["label"] = -1
    else:
        df["label"] = df["label"].fillna(-1).astype(int)
    if "center" not in df.columns:
        df["center"] = "unknown"
    else:
        df["center"] = df["center"].fillna("unknown").astype(str)
    return df

# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE_STR if torch.cuda.is_available() and DEVICE_STR == "cuda" else "cpu")
    print(f"Using device: {device}")

    # 准备索引
    index_df = prepare_index(INDEX_CSV, IMAGE_ROOT)
    index_df.to_csv(output_dir / "input_index_resolved.csv", index=False)

    # 加载模型
    seg_model, seg_channels, seg_num_res_units = load_segmentation_model(SEG_MODEL_PATH, device)
    cls_model = load_classifier_model(CLS_MODEL_PATH, device)

    # 生成分割概率图
    prob_df = generate_probability_maps(index_df, seg_model, output_dir, device)
    if len(prob_df) == 0:
        raise RuntimeError("No probability maps generated.")

    # 构建 ROI 数据集
    roi_df, window_df, failed_df = build_roi_dataset(
        prob_df=prob_df,
        output_dir=output_dir,
        k=K,
        margin=ROI_MARGIN,
        roi_size=ROI_SIZE,
        seg_threshold=SEG_THRESHOLD,
        valid_area_ratio_threshold=VALID_AREA_RATIO_THRESHOLD,
    )
    if len(roi_df) == 0:
        raise RuntimeError("No ROI images generated.")

    # 切片级分类预测
    slice_pred = predict_slice_probabilities(
        model=cls_model,
        roi_df=roi_df,
        batch_size=BATCH_SIZE,
        roi_size=ROI_SIZE,
        device=device,
    )
    # 聚合为患者级预测
    patient_pred = aggregate_patient_probabilities(slice_pred, AGGREGATION)
    patient_pred = assign_prediction(patient_pred, THRESHOLD)

    # 保存结果
    slice_pred.to_csv(output_dir / "slice_predictions.csv", index=False)
    patient_pred.to_csv(output_dir / "patient_predictions.csv", index=False)
 
    config = {
        "index_csv": INDEX_CSV,
        "image_root": IMAGE_ROOT,
        "seg_model": SEG_MODEL_PATH,
        "cls_model": CLS_MODEL_PATH,
        "threshold": THRESHOLD,
        "k": K,
        "roi_margin": ROI_MARGIN,
        "aggregation": AGGREGATION,
        "seg_threshold": SEG_THRESHOLD,
        "valid_area_ratio_threshold": VALID_AREA_RATIO_THRESHOLD,
        "roi_size": ROI_SIZE,
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "seed": SEED,
        "segmentation_channels": list(seg_channels),
        "segmentation_num_res_units": seg_num_res_units,
    }
    with open(output_dir / "inference_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nPatient-level predictions:")
    print(patient_pred[["pid", "label", "prob", "pred_label", "prediction"]])

if __name__ == "__main__":
    main()
