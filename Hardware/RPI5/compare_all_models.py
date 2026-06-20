"""
VeriPass - 7개 모델 종합 비교표 (분석 ②, 최종판)
================================================
동일 test set(1885장) + 3-class 지표로 7개 모델을 비교한다.
백본 3계열(InsightFace / MiVOLO v1-224 / MiVOLO v2-384) × (base·fine-tuned) + 경량 CNN.

결정 기준: ① gender ② age MAE ③ teen/senior recall ④ RPi 추론시간 ⑤ 안정성

RPi 추론시간(초)은 RPi5 실측치(사용자 제공): InsightFace 0.7 / MiVOLO v1 3.3 / MiVOLO v2 1.5.
CNN 은 미측정(가장 경량이라 가장 빠를 것으로 추정).
gender/age MAE: InsightFace FT 와 MiVOLO FT 는 백본 값(=각 base 와 동일). 파인튜닝은 3-class 만 개선.

실행: python compare_all_models.py
"""

import json

# (라벨, json, RPi 추론시간[초] or None)
# RPi 추론시간(초): base 모델만 RPi5 실측. 파인튜닝(FT) 모델 + CNN 은 미측정(None) -> '-'.
MODELS = [
    ("CNN",         "01_CNN/cnn_metrics.json",                    None),
    ("IF-base",     "02_InsightFace_base/baseline_insight_metrics.json",       0.7),
    ("IF-FT",       "03_InsightFace_finetuned/finetuned_insight_metrics.json",      None),
    ("MV1-224", "04_MiVOLO_v1_224/mivolo_facebody_metrics.json", 3.3),
    ("MV1-224-FT",  "05_MiVOLO_v1_224_finetuned/mivolo_v1_224_finetuned_metrics.json", None),
    ("MV2-384",     "06_MiVOLO_v2_384/mivolo_v2_384_metrics.json",          1.5),
    ("MV2-384-FT",  "07_MiVOLO_v2_384_finetuned/mivolo_v2_384_finetuned_metrics.json", None),
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
    data = [(name, load(path), rpi) for name, path, rpi in MODELS]
    names = [n for n, _, _ in data]

    rows = [
        ("gender acc ★①",   lambda d, r: g(d, "gender_accuracy"), 3, True),
        ("age MAE ★②",      lambda d, r: g(d, "age_mae_overall"), 2, False),
        ("3-class acc",      lambda d, r: g(d, "three_class_accuracy"), 3, True),
        ("macro-F1",         lambda d, r: g(d, "macro_f1"), 3, True),
        ("teen recall ★③",  lambda d, r: g(d, "per_class", "teen_13_18", "recall"), 3, True),
        ("teen prec",        lambda d, r: g(d, "per_class", "teen_13_18", "precision"), 3, True),
        ("senior recall ★③",lambda d, r: g(d, "per_class", "senior_65plus", "recall"), 3, True),
        ("senior prec",      lambda d, r: g(d, "per_class", "senior_65plus", "precision"), 3, True),
        ("RPi 추론(초) ★④", lambda d, r: r, 1, False),
        ("4090 FPS",         lambda d, r: g(d, "fps"), 0, True),
    ]

    colw = 12
    header = f"{'metric':<16}" + "".join(f"{n:>{colw}}" for n in names)
    print(header)
    print("-" * len(header))
    for label, fn, nd, higher in rows:
        vals = [fn(d, r) for _, d, r in data]
        cells = [fmt(v, nd) for v in vals]
        nums = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float))]
        best_i = None
        if higher is not None and nums:
            best_i = (max(nums, key=lambda t: t[1]) if higher else min(nums, key=lambda t: t[1]))[0]
        line = f"{label:<16}"
        for i, c in enumerate(cells):
            line += f"{(c + ('*' if i == best_i else ' ')):>{colw}}"
        print(line)

    print("\n범례: * = 해당 지표 최우수 | ★+번호 = 결정 기준 우선순위")
    print("해석:")
    print(" - MV2-384(base) 가 gender/age MAE/macro-F1/3-class acc 전부 1위이며 RPi 1.5초로 배포 가능 → 종합 최강.")
    print(" - 파인튜닝(IF/MV1/MV2 FT)은 teen/senior recall 을 크게 올리지만 precision 이 떨어져 macro-F1 은 하락(class-weighted).")
    print("   => '취약계층을 절대 놓치지 않기(recall)' 가 최우선이면 FT, '균형/오탐 최소'가 중요하면 base.")
    print(" - gender/age MAE 가 같은 칸(IF-base=IF-FT, MV1=MV1-FT, MV2=MV2-FT)은 백본 공유 때문(파인튜닝은 3-class 만 변경).")
    print(" - MV1(224) 은 RPi 3.3초로 사실상 배포권 밖. CNN 은 최경량이나 teen recall 0.31 로 청소년 검증 취약.")


if __name__ == "__main__":
    main()
