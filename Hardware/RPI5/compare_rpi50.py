"""
VeriPass - 7개 모델 RPi5 비교표 
=================================================
evaluate_rpi_all.py 로 생성된 *_rpi.json 을 읽어 비교표 출력.
(PC 결과는 compare_all_models.py 참고)

실행: python compare_rpi50.py
"""

import json

# RPi 추론시간: mean_latency_ms/1000 (JSON에서 자동 읽음)
# ※ MV1/MV2 는 YOLO 검출 시간 미포함 (backbone 순수 추론시간만)
MODELS = [
    ("CNN",      "01_CNN/cnn_metrics_rpi50.json",                                   ),
    ("IF-base",  "02_InsightFace_base/baseline_insight_metrics_rpi.json",         ),
    ("IF-FT",    "03_InsightFace_finetuned/finetuned_insight_metrics_rpi50.json",   ),
    ("MV1-224",  "04_MiVOLO_v1_224/mivolo_facebody_metrics_rpi50.json",             ),
    ("MV1-FT",   "05_MiVOLO_v1_224_finetuned/mivolo_v1_224_finetuned_metrics_rpi50.json", ),
    ("MV2-384",  "06_MiVOLO_v2_384/mivolo_v2_384_metrics_rpi50.json",              ),
    ("MV2-FT",   "07_MiVOLO_v2_384_finetuned/mivolo_v2_384_finetuned_metrics_rpi50.json", ),
]


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def fmt(v, nd):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    data  = [(name, load(path)) for name, path in MODELS]
    names = [n for n, _ in data]

    rows = [
        ("gender acc ★①",    lambda d: g(d, "gender_accuracy"),                          3, True),
        ("age MAE ★②",       lambda d: g(d, "age_mae_overall"),                          2, False),
        ("3-class acc",       lambda d: g(d, "three_class_accuracy"),                     3, True),
        ("macro-F1",          lambda d: g(d, "macro_f1"),                                 3, True),
        ("teen recall ★③",   lambda d: g(d, "per_class", "teen_13_18", "recall"),        3, True),
        ("teen prec",         lambda d: g(d, "per_class", "teen_13_18", "precision"),     3, True),
        ("senior recall ★③", lambda d: g(d, "per_class", "senior_65plus", "recall"),     3, True),
        ("senior prec",       lambda d: g(d, "per_class", "senior_65plus", "precision"),  3, True),
        ("ms/img ★④",        lambda d: g(d, "mean_latency_ms"),                          0, False),
        ("FPS",               lambda d: g(d, "fps"),                                      1, True),
    ]

    colw = 12
    header = f"{'metric':<18}" + "".join(f"{n:>{colw}}" for n in names)
    print(f"\n{'='*len(header)}")
    print(f"RPi5 실측 비교표  (50장, *_rpi50.json)")
    print(header)
    print("-" * len(header))
    for label, fn, nd, higher in rows:
        vals  = [fn(d) for _, d in data]
        cells = [fmt(v, nd) for v in vals]
        nums  = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float))]
        best_i = None
        if higher is not None and nums:
            best_i = (max(nums, key=lambda t: t[1]) if higher else min(nums, key=lambda t: t[1]))[0]
        line = f"{label:<18}"
        for i, c in enumerate(cells):
            line += f"{(c + ('*' if i == best_i else ' ')):>{colw}}"
        print(line)

    print("=" * len(header))
    print("\n범례: * = 해당 지표 최우수 | ★ = 결정 기준")
    print("주의: ms/img 는 backbone 추론 시간. MV1/MV2 는 YOLO 검출시간 미포함.")
    print("      (실제 MV1 총 추론 ~3.8s/img, MV2 총 추론 ~6.5s/img 포함 YOLO)")


if __name__ == "__main__":
    main()
