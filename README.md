# VeriPass

**부정승차 단속 역무원 업무 과중 해결 및 정당한 대중교통 문화 만들기 — QR 스캔 + AI 얼굴 분석으로 우대권 부정사용을 실시간 자동 탐지합니다.**

<br>

## 프로젝트 소개

VeriPass는 지하철 우대 교통카드(청소년 우대권 · 경로 우대권) 부정승차를 자동으로 탐지하는 AI 기반 스마트 개찰구 프로토타입입니다.

사용자가 QR 코드를 개찰구에 스캔하면, RPi5 카메라가 탑승자의 얼굴을 분석해 나이대·성별을 추정하고 QR에 등록된 정보와 즉시 비교합니다. 일치하면 서보모터 게이트가 열리고, 불일치하면 게이트는 잠금 상태를 유지하며 대시보드에 실시간 경고 알림이 발생합니다.

**개발 기간** : 2026년 2월 18일 ~ 2026년 2월 19일

<br>



## 팀 구성

| 이름 | 역할 |
|------|------|
| 정가은 | PM · 데이터셋 구축 · AI 모델 학습 및 검증 · 웹 대시보드 개발 |
| 이다영 | RPi5 모델 검증 · HW 개발 (RPi4 QR 스캐너 · RPi5 AI 추론) |

<br>



## 주요 기능

- **QR 코드 스캔** : RPi4 + Camera Module v2 + pyzbar로 사용자 정보(name · age · gender · card_type · gate) 인식
- **AI 얼굴 분석** : RPi5 + Camera Module v3으로 탑승자 얼굴을 실시간 캡처 → InsightFace fine-tuned 모델로 연령대·성별 추정
- **실시간 판정** : QR 등록 정보와 AI 추정 결과 비교 → `normal` / `suspect` 판정
- **서보모터 게이트 제어** : normal → 서보 90° 회전(개방) / suspect → 0° 유지(잠금)
- **Firebase 실시간 연동** : RPi4 ↔ Firebase ↔ RPi5 삼각 구조로 데이터 실시간 동기화
- **관제 대시보드** : 부정승차 의심 발생 시 팝업 알림 · 게이트별 현황 · 누적 과징금 조회
- **세션 로그 기록** : `session_N` 키로 전체 스캔 이력 순서 보장 · Firebase 저장

<br>



## 기술 스택

![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi%205-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi%204-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=flat-square&logo=firebase&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

| 분류 | 기술 |
|------|------|
| QR 스캐너 디바이스 | Raspberry Pi 4 + Camera Module v2 |
| AI 추론 디바이스 | Raspberry Pi 5 + Camera Module v3 |
| QR 인식 | Camera Module v2 + pyzbar |
| AI 모델 | InsightFace buffalo_l (fine-tuned) |
| 데이터셋 | AAF + UTKFace Asian subset (12,561장, 3-class) |
| 게이트 제어 | SG90 서보모터 + RPi GPIO PWM |
| 백엔드 DB | Firebase Realtime DB |
| 웹 스트리밍 | Flask (port 5000) |
| 대시보드 | Web (실시간 모니터링 · 과징금 관리) |

<br>
<br>



## 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | AAF (All-Age-Faces, Tsinghua Univ.) + UTKFace Asian subset |
| 원본 규모 | 14,875장 |
| 최종 규모 | 12,561장 (0~12세 영유아 제외) |
| 클래스 | `teen_13_18` (청소년 우대권) / `adult_19_64` (일반) / `senior_65plus` (경로 우대권) |
| Split | Stratified 70 / 15 / 15 (train 8,792 / val 1,884 / test 1,885) |

> AIHub 한국인 안면 데이터는 승인 대기로 미반영. 한국인과 유사한 Asian 공개 데이터셋으로 대체했습니다.

<br>


## AI 모델 선정

총 **7개 모델**을 동일한 test set(1,885장)으로 비교 평가했습니다.

| 모델 | macro-F1 | 청소년 recall | 고령자 recall | RPi5 추론시간 |
|------|----------|--------------|--------------|--------------|
| CNN (MobileNetV3) | 0.667 | 0.309 | 0.631 | 0.1초 |
| InsightFace baseline | 0.548 | 0.113 | 0.683 | 0.7초 |
| **InsightFace fine-tuned ✅** | **0.746** | **0.651** | **0.740** | **~0.7초** |
| MiVOLO v1 base | 0.700 | 0.577 | 0.621 | 3.3초 |
| MiVOLO v1 FT | 0.694 | 0.724 | 0.929 | 0.5초 |
| MiVOLO v2 base | 0.782 | 0.618 | 0.780 | 1.5초 |
| MiVOLO v2 FT | 0.718 | 0.846 | 0.929 | 3.3초 |

> **최종 선정: InsightFace fine-tuned**
> VeriPass는 청소년·고령자 우대권을 **둘 다** 검증해야 합니다. CNN은 청소년 recall 0.31로 청소년 카드 검증이 사실상 불가능하고, MiVOLO v2는 성능은 우수하나 RPi5에서 1.5초로 IF-FT(~0.7초)보다 느립니다. IF-FT는 macro-F1·청소년/고령자 recall 모두 균형 있게 높고, 약점(고령자 precision)은 임계값 조정으로 보완 가능합니다.



## 시스템 동작 흐름

```
사용자 QR 제시
     │
     ▼
[RPi4] Camera Module v2 + pyzbar로 QR 스캔
     │  (name · age · gender · card_type · gate)
     ▼
Firebase Realtime DB 업로드
(/veripass/logs/session_N · /veripass/gates/{gate}/scan_result)
     │
     ▼
[RPi5] Firebase 리스닝 → 카메라 캡처 → InsightFace 추론
     │  (나이대 · 성별 추정)
     ▼
QR 정보 vs AI 추정 비교
     │
     ├─ 일치 → Status: normal → 서보 90° 개방 → 대시보드 정상 로그
     │
     └─ 불일치 → Status: suspect → 서보 0° 유지 → 대시보드 경고 알림
```

<br>



## 프로젝트 구조

```
VeriPass/
├── Hardware/
│   ├── RPI4/                          # RPi4 — QR 스캐너
│   │   ├── generator_qr.py            # QR 코드 생성기
│   │   ├── rpi4_scanner.py            # Camera Module v2 + pyzbar QR 스캔 · Firebase 전송 · 서보 제어
│   │   └── requirements.txt
│   │
│   └── RPI5/                          # RPi5 — AI 추론 · 모델 검증
│       ├── graphs/                    # 모델 비교 그래프 출력 폴더
│       ├── realtime_detect.py         # 카메라 실시간 얼굴 인식 · 성별·연령 추정
│       ├── compare_all_models.py      # 7개 모델 전체 비교 평가
│       ├── compare_rpi50.py           # RPi5 환경 성능 비교
│       ├── evaluate_rpi_all.py        # RPi5 실측 추론시간 · 지표 평가
│       ├── make_graphs_rpi.py         # 평가 결과 그래프 생성
│       ├── NOTION_REPORT.md           # 모델 비교 분석 리포트
│       └── requirements.txt
│
└── Software/
    ├── Model/                         # AI 모델 학습 · 검증
    │   └── (InsightFace fine-tuning · 데이터셋 전처리 · 학습 스크립트)
    │
    └── Web/                           # 웹 관제 대시보드
        ├── fonts/
        ├── images/
        ├── Main.html                  # 실시간 모니터링 · 게이트 현황 · 경고 팝업
        ├── Cards.html                 # 카드별 스캔 이력 조회
        ├── Fines.html                 # 누적 과징금 관리
        ├── app.js                     # Firebase 연동 · 실시간 데이터 처리
        ├── cards.js                   # 카드 이력 렌더링
        ├── fines.js                   # 과징금 계산 · 렌더링
        ├── firebase.js                # Firebase 초기화 · DB 참조
        ├── icons.js                   # 아이콘 유틸리티
        ├── style.css                  # 전체 스타일
        └── data.json                  # 로컬 테스트용 데이터
```

<br>




## 설치 및 실행

### 공통 사전 준비

Firebase 서비스 계정 키를 각 디바이스의 실행 디렉토리에 위치시킵니다.

> `picamera2`는 pip 패키지가 아닌 RPi OS 시스템 패키지이므로 아래 명령으로 별도 설치합니다.
> ```bash
> sudo apt install -y python3-picamera2
> ```

---

### RPi4 — QR 스캐너 실행

```bash
cd Hardware/RPI4

# 가상환경 생성 (picamera2 시스템 패키지 접근을 위해 --system-site-packages 사용)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 실행
python rpi4_scanner.py
# Flask 스트리밍: http://<RPi4_IP>:5000
```

---

### RPi5 — 실시간 AI 추론 실행

```bash
cd Hardware/RPI5

# 가상환경 생성
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 실행
python realtime_detect.py       # 카메라 실시간 얼굴 인식 · 판정
```

### RPi5 — 모델 성능 평가

```bash
# 가상환경이 활성화된 상태에서 실행 (위 단계 완료 후)
python evaluate_rpi_all.py      # 전체 모델 RPi5 실측 평가
python compare_all_models.py    # 7개 모델 비교
python make_graphs_rpi.py       # 결과 그래프 생성 → graphs/ 저장
```

### 웹 대시보드

```bash
cd Software/Web
# firebase.js 내 Firebase 설정값 입력 후 Main.html 브라우저에서 열기
```


<br>

---

## 판정 결과

| 판정 | 조건 | 게이트 | 대시보드 |
|------|------|--------|---------|
| `normal` | 나이대 · 성별 모두 일치 | 서보 90° 개방 | 정상 로그 기록 |
| `suspect` | 나이대 또는 성별 불일치 | 서보 0° 유지 (잠금) | 경고 팝업 알림 발생 |

<br>


