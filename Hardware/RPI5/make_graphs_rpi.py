"""
VeriPass - (RPi용) 전체 그래프 생성 + RPi 추론속도 포함
=======================================================
PC용 make_graphs.py 와 동일한 그래프(분류지표/나이MAE/CNN 학습곡선)에
**RPi5 추론시간**까지 추가로 그린다.

사용법(RPi):
  1) 아래 RPI_INFERENCE_SEC 에 7개 모델의 RPi 추론시간(초)을 채운다. (None = 미측정 → 그래프 제외)
  2) python make_graphs_rpi.py
  3) graphs/ 폴더에 결과 저장.

metrics JSON 은 각 모델 폴더(01_CNN/ ...) 또는 현재 폴더(flat) 어디서든 자동 탐색.
폰트: 번들된 fonts/Pretendard → 없으면 NanumGothic → 없으면 기본.

색상 규칙(절대 준수): 모델 고정색, 옐로우(#F1FF5E)는 종합 최우수(MV2-384) 1개에만.

출력(graphs/):
  00_overall.png            : 3패널 (분류지표 / 나이 MAE / RPi 추론시간)
  01~06                     : 지표별 개별 막대
  07_cnn_training_curve.png : CNN 학습 곡선 (training_history.csv 있을 때)
  08_rpi_inference.png      : RPi5 추론시간 (측정된 모델만)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ============================================================
# RPi5 추론시간(초) — evaluate_rpi_all.py 실측값 (mean_latency_ms/1000)
# ※ MV1/MV2 는 backbone 순수 추론시간. YOLO 검출 미포함(MV1 실제 ~3.8s, MV2 ~6.5s)
RPI_INFERENCE_SEC = {
    "CNN":     0.064,   # 63.5ms  (backbone only, 얼굴 검출 없음)
    "IF-base": 0.797,   # 796.5ms (InsightFace 전체 파이프라인)
    "IF-FT":   0.803,   # 803.3ms (InsightFace + MLP head)
    "MV1-224": 0.475,   # 475.2ms (backbone only, YOLO 미포함)
    "MV1-FT":  0.476,   # 475.9ms (backbone + head, YOLO 미포함)
    "MV2-FT":  3.270,   # 3270ms  (v2 base + FT features, YOLO 미포함)
    "MV2-384": 3.269,   # 3268.9ms (v2 base + FT features, YOLO 미포함)
}
# ============================================================

THIS = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS, "graphs")
EDGE = "#333333"

# (라벨, json파일명, 모델폴더, 고정색) 순서 A..G, G=MV2-384=최우수=옐로우
MODELS = [
    ("CNN",     "cnn_metrics_rpi50.json",                     "01_CNN",                     "#006AFF"),
    ("IF-base", "baseline_insight_metrics_rpi50.json",        "02_InsightFace_base",        "#004EC3"),
    ("IF-FT",   "finetuned_insight_metrics_rpi50.json",       "03_InsightFace_finetuned",   "#3A9AFF"),
    ("MV1-224", "mivolo_facebody_metrics_rpi50.json",         "04_MiVOLO_v1_224",           "#98C2FF"),
    ("MV1-FT",  "mivolo_v1_224_finetuned_metrics_rpi50.json", "05_MiVOLO_v1_224_finetuned", "#E5EDFF"),
    ("MV2-FT",  "mivolo_v2_384_finetuned_metrics_rpi50.json", "07_MiVOLO_v2_384_finetuned", "#777777"),
    ("MV2-384", "mivolo_v2_384_metrics_rpi50.json",           "06_MiVOLO_v2_384",           "#F1FF5E"),
]

CNN_HISTORY_CANDIDATES = [
    os.path.join(THIS, "training_history.csv"),
    os.path.join(THIS, "01_CNN", "training_history.csv"),
    os.path.join(THIS, "..", "outputs", "lightweight_cnn", "training_history.csv"),
]


def setup_font():
    # 1) 번들 Pretendard
    for w in ["Regular", "Bold"]:
        p = os.path.join(THIS, "fonts", f"Pretendard-{w}.ttf")
        if os.path.exists(p):
            fm.fontManager.addfont(p)
    names = {f.name for f in fm.fontManager.ttflist}
    if "Pretendard" in names:
        plt.rcParams["font.family"] = "Pretendard"
    else:
        # 2) NanumGothic (RPi 흔함) 3) 기본
        nanum = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(nanum):
            fm.fontManager.addfont(nanum)
            plt.rcParams["font.family"] = fm.FontProperties(fname=nanum).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    print("[font]", plt.rcParams["font.family"])


def resolve_json(fname, subfolder):
    for cand in [os.path.join(THIS, subfolder, fname), os.path.join(THIS, fname)]:
        if os.path.exists(cand):
            return cand
    return None


def load(fname, subfolder):
    p = resolve_json(fname, subfolder)
    if p is None:
        print("  [경고] metrics 못 찾음:", fname)
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def g(d, *keys, default=np.nan):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def main():
    setup_font()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = [(n, load(fn, sf), c) for n, fn, sf, c in MODELS]
    labels = [n for n, _, _ in data]
    colors = [c for _, _, c in data]
    x = np.arange(len(data))

    macro  = [g(d, "macro_f1") for _, d, _ in data]
    teen_r = [g(d, "per_class", "teen_13_18", "recall") for _, d, _ in data]
    sen_r  = [g(d, "per_class", "senior_65plus", "recall") for _, d, _ in data]
    gender = [g(d, "gender_accuracy") for _, d, _ in data]
    acc3   = [g(d, "three_class_accuracy") for _, d, _ in data]
    mae    = [g(d, "age_mae_overall") for _, d, _ in data]
    rpi    = [RPI_INFERENCE_SEC.get(n, None) for n, _, _ in data]
    rpi_v  = [r if r is not None else np.nan for r in rpi]
    has_rpi = any(r is not None for r in rpi)

    def labeled_bar(ax, vals, fmt, fs=8):
        b = ax.bar(x, vals, 0.64, color=colors, edgecolor=EDGE, linewidth=0.5)
        ax.bar_label(b, labels=[("" if (v is None or (isinstance(v, float) and np.isnan(v))) else fmt % v)
                                for v in vals], padding=2, fontsize=fs)

    def bar_single(values, title, ylabel, fname, fmt="%.3f", ymax=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        labeled_bar(ax, values, fmt, fs=9)
        finite = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
        ax.set_ylim(0, ymax if ymax else (max(finite) * 1.2 if finite else 1))
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, fname), dpi=150); plt.close(fig)
        print("  saved:", fname)

    # ---- 00 종합 (3패널: 분류 / 나이 MAE / RPi) ----
    ncol = 3 if has_rpi else 2
    widths = [2.4, 1.2, 1.2] if has_rpi else [2.4, 1.3]
    fig, axes = plt.subplots(1, ncol, figsize=(7.5 * ncol + 1, 6.6),
                             gridspec_kw={"width_ratios": widths})
    axA = axes[0]
    metric_defs = [("macro-F1", macro), ("청소년 recall ★", teen_r),
                   ("고령자 recall ★", sen_r), ("성별 정확도 ★", gender)]
    xm = np.arange(len(metric_defs)); width = 0.84 / len(data)
    for mi, (name, _, color) in enumerate(data):
        vals = [md[1][mi] for md in metric_defs]
        axA.bar(xm + (mi - (len(data) - 1) / 2) * width, vals, width,
                label=name, color=color, edgecolor=EDGE, linewidth=0.5)
    axA.set_ylim(0, 1.05); axA.set_ylabel("점수 (높을수록 좋음)")
    axA.set_title("(A) 분류 핵심 지표 — 부정승차 탐지", fontweight="bold")
    axA.set_xticks(xm); axA.set_xticklabels([m[0] for m in metric_defs])
    axA.legend(fontsize=9, ncol=4, loc="lower center"); axA.grid(axis="y", linestyle="--", alpha=0.35)

    axB = axes[1]
    labeled_bar(axB, mae, "%.1f")
    fin = [v for v in mae if not np.isnan(v)]
    axB.set_ylim(0, (max(fin) * 1.28 if fin else 10))
    axB.set_ylabel("나이 MAE (년, ↓ 좋음) ★②"); axB.set_title("(B) 나이 추정 오차", fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axB.grid(axis="y", linestyle="--", alpha=0.35)

    if has_rpi:
        axC = axes[2]
        labeled_bar(axC, rpi_v, "%.1f s")
        fin = [v for v in rpi_v if not np.isnan(v)]
        axC.set_ylim(0, (max(fin) * 1.3 if fin else 4))
        axC.set_ylabel("RPi5 추론시간 (초, ↓ 좋음) ★④"); axC.set_title("(C) RPi5 추론시간", fontweight="bold")
        axC.set_xticks(x); axC.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        axC.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("VeriPass: 7개 모델 종합 비교 (RPi5) · 옐로우=종합 최우수 MV2-384",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "00_overall.png"), dpi=150); plt.close(fig)
    print("  saved: 00_overall.png")

    # ---- 01~06 개별 ----
    bar_single(macro,  "macro-F1 (클래스 균형 종합지표)", "macro-F1", "01_macro_f1.png", ymax=1.0)
    bar_single(teen_r, "청소년 recall ★", "recall", "02_teen_recall.png", ymax=1.0)
    bar_single(sen_r,  "고령자 recall ★", "recall", "03_senior_recall.png", ymax=1.0)
    bar_single(gender, "성별 정확도 ★①", "accuracy", "04_gender_accuracy.png", ymax=1.05)
    bar_single(acc3,   "3-클래스 정확도", "accuracy", "05_3class_accuracy.png", ymax=1.0)
    bar_single(mae,    "나이 MAE ★② (낮을수록 좋음)", "MAE (년)", "06_age_mae.png", fmt="%.2f")

    # ---- 07 CNN 학습 곡선 ----
    hist = next((c for c in CNN_HISTORY_CANDIDATES if os.path.exists(c)), None)
    if hist:
        import pandas as pd
        h = pd.read_csv(hist)
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.plot(h["epoch"], h["train_loss"], color="#006AFF", marker="o", ms=3, label="train_loss")
        ax1.plot(h["epoch"], h["val_loss"], color="#004EC3", marker="s", ms=3, label="val_loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.grid(True, linestyle="--", alpha=0.35)
        ax2 = ax1.twinx()
        ax2.plot(h["epoch"], h["val_mae"], color="#777777", marker="^", ms=3, ls="--", label="val_MAE(년)")
        ax2.set_ylabel("val MAE (년)")
        l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=9)
        ax1.set_title("경량 CNN(MobileNetV3) 학습 곡선", fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "07_cnn_training_curve.png"), dpi=150); plt.close(fig)
        print("  saved: 07_cnn_training_curve.png")

    # ---- 08 RPi 추론시간 (측정된 모델만) ----
    if has_rpi:
        bar_single(rpi_v, "RPi5 추론시간 ★④ (낮을수록 좋음)", "초", "08_rpi_inference.png", fmt="%.1f")

    print(f"\n완료 -> {OUT_DIR}/  (RPi 추론시간 {'포함' if has_rpi else '미입력→제외'})")


if __name__ == "__main__":
    main()
