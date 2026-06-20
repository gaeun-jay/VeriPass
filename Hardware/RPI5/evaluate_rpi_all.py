#!/usr/bin/env python3
"""
VeriPass - RPi 7개 모델 통합 평가 (처음 TOP_N장)
=================================================
rpi_validation_package/validation_set.csv 
7개 모델을 모두 평가하고 각 모델 폴더에 *_rpi50.json 저장.
실행 후 비교표 출력.

사용법:
    python evaluate_rpi_all.py
"""

import os, sys, json, time, importlib.util, types as _types
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

# ===== PATHS =====
VERIPASS    = "/home/rpi/VeriPass"
MIVOLO_DIR  = "/home/rpi/MiVOLO"
V2_DIR      = os.path.join(MIVOLO_DIR, "models")
CSV_PATH    = os.path.join(VERIPASS, "rpi_validation_package/validation_set.csv")
IMG_BASE    = os.path.join(VERIPASS, "rpi_validation_package")
YOLO_PT     = os.path.join(V2_DIR, "yolov8x_person_face.pt")
MV1_CKPT    = os.path.join(V2_DIR, "model_imdb_cross_person_4.22_99.46.pth.tar")
CNN_PT      = os.path.join(VERIPASS, "01_CNN/best_model.pt")
IF_HEAD_PT  = os.path.join(VERIPASS, "veripass_insight_head.pt")
MV1_HEAD_PT = os.path.join(VERIPASS, "05_MiVOLO_v1_224_finetuned/veripass_mivolo_v1_224_finetuned_head.pt")
MV2_HEAD_PT = os.path.join(VERIPASS, "07_MiVOLO_v2_384_finetuned/veripass_mivolo_v2_384_finetuned_head.pt")

TOP_N   = 50
DEVICE  = torch.device("cpu")
CLASSES = ["teen_13_18", "adult_19_64", "senior_65plus"]

# rpi_validation_package CSV 성별 인코딩: 0=Male, 1=Female (UTKFace/AAF 원본 convention)
# test_code.py 주석(0=female,1=male)은 오기 — UTKFace 파일명으로 검증: gender=1→Female
GENDER_MAP  = {0: "M", 1: "F", "0": "M", "1": "F"}
# MiVOLO v2 config gender_id2label: 0=male, 1=female
V2_GENDER   = {0: "M", 1: "F"}
# CNN gender head 학습 인코딩: 0=Male, 1=Female
CNN_GENDER  = {0: "M", 1: "F"}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

sys.path.insert(0, MIVOLO_DIR)


# ===== 공통 유틸 =====
def age_to_class(age):
    if age < 19:  return "teen_13_18"
    if age < 65:  return "adult_19_64"
    return "senior_65plus"


def prf_from_confusion(confusion, classes):
    per_class, f1s = {}, []
    for c in classes:
        tp = confusion[c][c]
        fn = sum(confusion[c][p] for p in classes) - tp
        fp = sum(confusion[t][c] for t in classes) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": tp + fn}
        f1s.append(f1)
    return per_class, round(sum(f1s) / len(f1s), 4)


def build_metrics(label, n_total, results):
    processed = sum(1 for r in results if not r.get("missing"))
    n_missing = sum(1 for r in results if r.get("missing"))
    n_no_face = sum(1 for r in results if r.get("no_face"))
    times     = [r["infer_time"] for r in results if not r.get("missing") and r.get("infer_time")]
    valid     = [r for r in results if not r.get("missing") and not r.get("no_face")
                 and r.get("pred_age") is not None]

    g_ok  = sum(1 for r in valid if r.get("pred_sex") and r.get("pred_sex") == r.get("true_sex"))
    g_tot = sum(1 for r in valid if r.get("pred_sex") and r.get("true_sex"))

    age_err = [abs(r["pred_age"] - r["true_age"]) for r in valid if r.get("true_age") is not None]
    pc_err  = {c: [] for c in CLASSES}
    for r in valid:
        tc = r.get("true_class")
        if tc in pc_err and r.get("true_age") is not None:
            pc_err[tc].append(abs(r["pred_age"] - r["true_age"]))

    conf = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    cls_ok = cls_tot = 0
    for r in valid:
        tc, pc = r.get("true_class"), r.get("pred_class")
        if tc in CLASSES and pc in CLASSES:
            conf[tc][pc] += 1; cls_tot += 1
            if tc == pc: cls_ok += 1

    fps     = processed / sum(times) if times else 0.0
    mean_ms = sum(times) / len(times) * 1000 if times else 0.0

    m = {
        "model": label, "device": "cpu (RPi5)",
        "test_images": n_total, "processed": processed,
        "missing_image_files": n_missing, "no_face_count": n_no_face,
        "fps": round(fps, 2), "mean_latency_ms": round(mean_ms, 1),
    }
    if g_tot:
        m["gender_accuracy"]   = round(g_ok / g_tot, 4)
        m["gender_eval_count"] = g_tot
    if age_err:
        m["age_mae_overall"]   = round(float(np.mean(age_err)), 3)
        m["age_mae_per_class"] = {c: round(float(np.mean(v)), 3) if v else None
                                   for c, v in pc_err.items()}
    if cls_tot:
        pc_m, mf1 = prf_from_confusion(conf, CLASSES)
        m["three_class_accuracy"]    = round(cls_ok / cls_tot, 4)
        m["macro_f1"]                = mf1
        m["per_class"]               = pc_m
        m["three_class_eval_count"]  = cls_tot
        m["three_class_confusion"]   = conf
    return m


def save_json(m, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"  -> {path}")


def load_df():
    df = pd.read_csv(CSV_PATH).head(TOP_N).reset_index(drop=True)
    print(f"CSV 로드: {len(df)}장  (columns: {list(df.columns)})")
    return df


def img_path(row):
    return os.path.join(IMG_BASE, row["filepath"])


def gt_info(row):
    true_age   = float(row["age"])
    true_sex   = GENDER_MAP.get(int(float(row["gender"])), None)
    true_class = str(row["age_class"]) if pd.notna(row.get("age_class", None)) else None
    return true_age, true_sex, true_class


# ===== YOLO 얼굴 검출 (MiVOLO 계열 공용) =====
_yolo = None
_face_cls_id = 1

def get_yolo():
    global _yolo, _face_cls_id
    if _yolo is None:
        print("  [YOLO] yolov8x_person_face 로드 중...")
        from ultralytics import YOLO
        _yolo = YOLO(YOLO_PT)
        names = _yolo.names if isinstance(_yolo.names, dict) else dict(enumerate(_yolo.names))
        _face_cls_id = next((k for k, v in names.items() if str(v).lower() == "face"), 1)
    return _yolo, _face_cls_id


def detect_face_crop(img_bgr):
    """YOLO로 가장 큰 face bbox를 크롭. 실패 시 원본 반환, face_detected=False."""
    yolo, face_cls = get_yolo()
    res   = yolo(img_bgr, verbose=False)[0]
    boxes = res.boxes
    best, best_area = None, -1.0
    if boxes is not None:
        for b in boxes:
            if int(b.cls[0]) != face_cls:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            a = (x2 - x1) * (y2 - y1)
            if a > best_area:
                best, best_area = (max(0, x1), max(0, y1), x2, y2), a
    if best:
        x1, y1, x2, y2 = best
        crop = img_bgr[y1:y2, x1:x2]
        return (crop if crop.size > 0 else img_bgr), True
    return img_bgr, False


# ===== MLP Head (InsightFace FT / MiVOLO FT 공용) =====
class MlpHead(nn.Module):
    def __init__(self, in_dim, hidden=256, n=3, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, n),
        )
    def forward(self, x):
        return self.net(x)


def load_head(path):
    ck = torch.load(path, map_location="cpu")
    head = MlpHead(in_dim=ck.get("in_dim", 384), hidden=ck.get("hidden", 256))
    head.load_state_dict(ck["state_dict"])
    head.to(DEVICE).eval()
    return head


# ===== InsightFace 얼굴 선택 =====
def pick_main_face(faces):
    best, ba = None, -1.0
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        a = (x2 - x1) * (y2 - y1)
        if a > ba:
            best, ba = f, a
    return best


# =============================================================
# MODEL 1: CNN (MobileNetV3-Large)
# =============================================================
class AgeGenderMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.mobilenet_v3_large(weights=None)
        self.backbone    = bb.features
        self.avgpool     = nn.AdaptiveAvgPool2d(1)
        self.age_head    = nn.Sequential(nn.Linear(960, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
        self.gender_head = nn.Sequential(nn.Linear(960, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, x):
        f = self.avgpool(self.backbone(x)).flatten(1)
        return self.age_head(f).squeeze(1), self.gender_head(f)


CNN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def eval_cnn(df):
    print("\n[1/7] CNN (MobileNetV3-Large)...")
    mdl = AgeGenderMobileNet().to(DEVICE)
    mdl.load_state_dict(torch.load(CNN_PT, map_location=DEVICE))
    mdl.eval()

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        true_age, true_sex, true_class = gt_info(row)
        try:
            pil = Image.open(img_path(row)).convert("RGB")
        except Exception:
            results.append({"missing": True, "true_age": true_age, "true_sex": true_sex, "true_class": true_class})
            continue
        x = CNN_TF(pil).unsqueeze(0).to(DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            age_t, g_t = mdl(x)
        elapsed = time.perf_counter() - t0

        pred_age = float(age_t.item())
        pred_sex = CNN_GENDER.get(int(g_t.argmax(1).item()))
        results.append({"missing": False, "no_face": False,
            "pred_age": pred_age, "pred_sex": pred_sex, "pred_class": age_to_class(pred_age),
            "true_age": true_age, "true_sex": true_sex, "true_class": true_class,
            "infer_time": elapsed})

    m = build_metrics("mobilenetv3_large_CNN (RPi5)", len(df), results)
    m["note"] = "whole-image 분류기 (얼굴 검출 없음)"
    save_json(m, os.path.join(VERIPASS, "01_CNN/cnn_metrics_rpi50.json"))
    return m


# =============================================================
# MODEL 2 & 3: InsightFace base / FT
# (InsightFace buffalo_l 한 번만 로드해서 두 모델 동시 평가)
# =============================================================
def eval_if_base_and_ft(df):
    print("\n[2/7] InsightFace base + [3/7] IF-FT 동시 평가...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    head = load_head(IF_HEAD_PT)

    res_base, res_ft = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="IF base+FT"):
        true_age, true_sex, true_class = gt_info(row)
        img = cv2.imread(img_path(row))
        if img is None:
            stub = {"missing": True, "true_age": true_age, "true_sex": true_sex, "true_class": true_class}
            res_base.append(stub); res_ft.append(stub); continue

        t0 = time.perf_counter()
        faces = app.get(img)
        if_time = time.perf_counter() - t0

        if not faces:
            stub = {"missing": False, "no_face": True, "infer_time": if_time,
                    "pred_age": None, "pred_sex": None, "pred_class": None,
                    "true_age": true_age, "true_sex": true_sex, "true_class": true_class}
            res_base.append(stub); res_ft.append(stub); continue

        face     = pick_main_face(faces)
        pred_age = float(face.age)
        pred_sex = face.sex  # 'M' or 'F'

        # base: 3-class는 backbone age 기반
        res_base.append({"missing": False, "no_face": False,
            "pred_age": pred_age, "pred_sex": pred_sex,
            "pred_class": age_to_class(pred_age),
            "true_age": true_age, "true_sex": true_sex, "true_class": true_class,
            "infer_time": if_time})

        # FT: 임베딩 → MLP head → 3-class
        emb = torch.tensor(face.normed_embedding.astype(np.float32), device=DEVICE).unsqueeze(0)
        t1 = time.perf_counter()
        with torch.no_grad():
            pred_idx = int(head(emb).argmax(1).item())
        ft_time = if_time + (time.perf_counter() - t1)

        res_ft.append({"missing": False, "no_face": False,
            "pred_age": pred_age, "pred_sex": pred_sex,
            "pred_class": CLASSES[pred_idx],
            "true_age": true_age, "true_sex": true_sex, "true_class": true_class,
            "infer_time": ft_time})

    m_base = build_metrics("buffalo_l_baseline (RPi5)", len(df), res_base)
    save_json(m_base, os.path.join(VERIPASS, "02_InsightFace_base/baseline_insight_metrics_rpi50.json"))

    m_ft = build_metrics("buffalo_l_finetuned (RPi5)", len(df), res_ft)
    m_ft["note"] = "gender/age MAE는 buffalo_l 백본값 (base와 동일). FT는 3-class head만 변경."
    save_json(m_ft, os.path.join(VERIPASS, "03_InsightFace_finetuned/finetuned_insight_metrics_rpi50.json"))
    return m_base, m_ft


# =============================================================
# MODEL 4 & 5: MiVOLO v1-224 base / FT
# (timm 백본 한 번만 로드해서 두 모델 동시 평가)
# =============================================================
def eval_mv1_base_and_ft(df):
    print("\n[4/7] MiVOLO v1-224 base + [5/7] MV1-FT 동시 평가...")
    from mivolo.model.mi_volo import MiVOLO
    from mivolo.data.misc import prepare_classification_images

    mv = MiVOLO(MV1_CKPT, "cpu", half=False, use_persons=True, disable_faces=False, verbose=False)
    timm_model = mv.model.float().eval()
    min_age, max_age, avg_age = mv.meta.min_age, mv.meta.max_age, mv.meta.avg_age
    head = load_head(MV1_HEAD_PT)

    res_base, res_ft = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MV1 base+FT"):
        true_age, true_sex, true_class = gt_info(row)
        img = cv2.imread(img_path(row))
        if img is None:
            stub = {"missing": True, "true_age": true_age, "true_sex": true_sex, "true_class": true_class}
            res_base.append(stub); res_ft.append(stub); continue

        crop, face_det = detect_face_crop(img)

        # 전처리: face 크롭 → (1,3,224,224) → 6ch (face+body 복사)
        faces_3ch = prepare_classification_images([crop], 224, MEAN, STD).to(DEVICE).float()
        inp_6ch   = torch.cat((faces_3ch, faces_3ch.clone()), dim=1)

        t0 = time.perf_counter()
        with torch.no_grad():
            # 한 번의 forward_features로 base와 FT 모두 계산
            feats  = timm_model.forward_features(inp_6ch)
            out    = timm_model.forward_head(feats)              # (1,3): gender*2 + age_norm
            pooled = timm_model.forward_head(feats, pre_logits=True)  # (1,384)
        elapsed = time.perf_counter() - t0

        # base 디코딩
        age_norm = float(out[0, 2].item())
        pred_age = age_norm * (max_age - min_age) + avg_age
        g_probs  = out[0, :2].softmax(0)
        pred_sex = "M" if float(g_probs[0]) > float(g_probs[1]) else "F"

        res_base.append({"missing": False, "no_face": not face_det,
            "pred_age": pred_age, "pred_sex": pred_sex, "pred_class": age_to_class(pred_age),
            "true_age": true_age, "true_sex": true_sex, "true_class": true_class,
            "infer_time": elapsed})

        # FT 디코딩 (head 추론 시간 추가)
        t1 = time.perf_counter()
        with torch.no_grad():
            pred_idx = int(head(pooled).argmax(1).item())
        ft_time = elapsed + (time.perf_counter() - t1)

        res_ft.append({"missing": False, "no_face": not face_det,
            "pred_age": pred_age, "pred_sex": pred_sex, "pred_class": CLASSES[pred_idx],
            "true_age": true_age, "true_sex": true_sex, "true_class": true_class,
            "infer_time": ft_time})

    m_base = build_metrics("mivolo_v1_224_base (RPi5)", len(df), res_base)
    save_json(m_base, os.path.join(VERIPASS, "04_MiVOLO_v1_224/mivolo_facebody_metrics_rpi50.json"))

    m_ft = build_metrics("mivolo_v1_224_finetuned (RPi5)", len(df), res_ft)
    m_ft["note"] = "gender/age MAE는 v1 백본값 (base와 동일). FT는 3-class head만 변경."
    save_json(m_ft, os.path.join(VERIPASS, "05_MiVOLO_v1_224_finetuned/mivolo_v1_224_finetuned_metrics_rpi50.json"))
    return m_base, m_ft


# =============================================================
# MODEL 6 & 7: MiVOLO v2-384 base / FT
# =============================================================
def load_v2_model():
    """importlib으로 v2 커스텀 모듈 직접 로드 (transformers Auto 우회)."""
    if "mivolo_models" not in sys.modules:
        pkg = _types.ModuleType("mivolo_models")
        pkg.__path__ = [V2_DIR]
        sys.modules["mivolo_models"] = pkg

    def lm(name, fpath):
        spec = importlib.util.spec_from_file_location(name, fpath)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "mivolo_models"
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_m  = lm("mivolo_models.configuration_mivolo",  os.path.join(V2_DIR, "configuration_mivolo.py"))
    mdl_m  = lm("mivolo_models.modeling_mivolo",        os.path.join(V2_DIR, "modeling_mivolo.py"))
    proc_m = lm("mivolo_models.mivolo_image_processor", os.path.join(V2_DIR, "mivolo_image_processor.py"))

    from safetensors.torch import load_file
    config = cfg_m.MiVOLOConfig.from_pretrained(V2_DIR)
    model  = mdl_m.MiVOLOForImageClassification(config)
    state  = load_file(os.path.join(V2_DIR, "model.safetensors"))
    model.load_state_dict(state, strict=False)
    model.float().eval()
    proc = proc_m.MiVOLOImageProcessor.from_pretrained(V2_DIR)
    return model, proc


def eval_mv2_base_and_ft(df):
    print("\n[6/7] MiVOLO v2-384 base + [7/7] MV2-FT 동시 평가...")
    hf_model, proc = load_v2_model()
    timm_model = hf_model.mivolo.model   # 내부 timm 백본
    head = load_head(MV2_HEAD_PT)

    res_base, res_ft = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MV2 base+FT"):
        true_age, true_sex, true_class = gt_info(row)
        img = cv2.imread(img_path(row))
        if img is None:
            stub = {"missing": True, "true_age": true_age, "true_sex": true_sex, "true_class": true_class}
            res_base.append(stub); res_ft.append(stub); continue

        crop, face_det = detect_face_crop(img)

        # MiVOLOImageProcessor: BGR→RGB, 384 리사이즈, normalize → (1,3,384,384)
        rgb = crop[:, :, ::-1].copy()
        inp_proc = proc(images=[rgb])
        px = inp_proc["pixel_values"].float()   # (1, 3, 384, 384)
        px_6ch = torch.cat((px, px.clone()), dim=1)  # (1, 6, 384, 384) for timm backbone

        t0 = time.perf_counter()
        with torch.no_grad():
            # base: HuggingFace 래퍼 사용 (faces+body 각각 3ch)
            out = hf_model(faces_input=px, body_input=px)
            # feature 추출: timm backbone 직접 사용 (6ch 입력)
            feats  = timm_model.forward_features(px_6ch)
            pooled = timm_model.forward_head(feats, pre_logits=True)  # (1, 384)
        elapsed = time.perf_counter() - t0

        pred_age = float(out.age_output[0])   # 이미 연수로 역정규화됨
        gidx     = int(out.gender_class_idx[0].item()) if out.gender_class_idx is not None else None
        pred_sex = V2_GENDER.get(gidx)

        res_base.append({"missing": False, "no_face": not face_det,
            "pred_age": pred_age, "pred_sex": pred_sex, "pred_class": age_to_class(pred_age),
            "true_age": true_age, "true_sex": true_sex, "true_class": true_class,
            "infer_time": elapsed})

        t1 = time.perf_counter()
        with torch.no_grad():
            pred_idx = int(head(pooled).argmax(1).item())
        ft_time = elapsed + (time.perf_counter() - t1)

        res_ft.append({"missing": False, "no_face": not face_det,
            "pred_age": pred_age, "pred_sex": pred_sex, "pred_class": CLASSES[pred_idx],
            "true_age": true_age, "true_sex": true_sex, "true_class": true_class,
            "infer_time": ft_time})

    m_base = build_metrics("mivolo_v2_384_base (RPi5)", len(df), res_base)
    save_json(m_base, os.path.join(VERIPASS, "06_MiVOLO_v2_384/mivolo_v2_384_metrics_rpi50.json"))

    m_ft = build_metrics("mivolo_v2_384_finetuned (RPi5)", len(df), res_ft)
    m_ft["note"] = "gender/age MAE는 v2 백본값 (base와 동일). FT는 3-class head만 변경."
    save_json(m_ft, os.path.join(VERIPASS, "07_MiVOLO_v2_384_finetuned/mivolo_v2_384_finetuned_metrics_rpi50.json"))
    return m_base, m_ft


# =============================================================
# 비교표 출력
# =============================================================
def print_compare(metrics_list, labels):
    def gv(m, *keys):
        d = m
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return None
        return d

    rows = [
        ("gender acc",  lambda m: gv(m, "gender_accuracy"),                          3, True),
        ("age MAE",     lambda m: gv(m, "age_mae_overall"),                          2, False),
        ("3-class acc", lambda m: gv(m, "three_class_accuracy"),                     3, True),
        ("macro-F1",    lambda m: gv(m, "macro_f1"),                                 3, True),
        ("teen recall", lambda m: gv(m, "per_class", "teen_13_18", "recall"),        3, True),
        ("senr recall", lambda m: gv(m, "per_class", "senior_65plus", "recall"),     3, True),
        ("ms/img (RPi)", lambda m: gv(m, "mean_latency_ms"),                         0, False),
    ]

    cw = 11
    hdr = f"{'metric':<14}" + "".join(f"{n:>{cw}}" for n in labels)
    sep = "=" * len(hdr)
    print(f"\n{sep}")
    print(f"RPi5 비교표 ({TOP_N}장)")
    print(hdr)
    print("-" * len(hdr))
    for label, fn, nd, higher in rows:
        vals  = [fn(m) for m in metrics_list]
        cells = [f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-" for v in vals]
        nums  = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float))]
        best_i = None
        if nums:
            best_i = (max(nums, key=lambda t: t[1]) if higher else min(nums, key=lambda t: t[1]))[0]
        line = f"{label:<14}"
        for i, c in enumerate(cells):
            line += f"{(c + ('*' if i == best_i else ' ')):>{cw}}"
        print(line)
    print(sep)
    print("* = 해당 지표 최우수")


# =============================================================
# MAIN
# =============================================================
def main():
    print("=" * 60)
    print(f"VeriPass RPi 통합 평가  (상위 {TOP_N}장)")
    print("=" * 60)

    df = load_df()

    m_cnn              = eval_cnn(df)
    m_if_base, m_if_ft = eval_if_base_and_ft(df)
    m_mv1, m_mv1_ft    = eval_mv1_base_and_ft(df)
    m_mv2, m_mv2_ft    = eval_mv2_base_and_ft(df)

    all_metrics = [m_cnn, m_if_base, m_if_ft, m_mv1, m_mv1_ft, m_mv2, m_mv2_ft]
    labels = ["CNN", "IF-base", "IF-FT", "MV1-224", "MV1-FT", "MV2-384", "MV2-FT"]
    print_compare(all_metrics, labels)
    print("\n완료. 각 모델 폴더에 *_rpi50.json 저장됨.")
    print("(compare_all_models.py 는 기존 PC JSON 그대로 유지)")


if __name__ == "__main__":
    main()
