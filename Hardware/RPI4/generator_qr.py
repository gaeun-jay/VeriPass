"""
QR코드 생성 — 카드 정보 전체를 JSON으로 QR에 인코딩
설치: pip install qrcode[pil] Pillow
실행: python generator_qr.py
"""

import qrcode, json
from PIL import Image, ImageDraw, ImageFont

cards = [
    {"card_id": "CARD_001", "name": "이다영", "age": 24, "gender": "female", "card_type": "adult",  "count": 0, "gate": 3},
    {"card_id": "CARD_002", "name": "이철수", "age": 70, "gender": "male",   "card_type": "senior", "count": 0, "gate": 3},
    {"card_id": "CARD_003", "name": "김민재", "age": 15, "gender": "male",   "card_type": "teen",   "count": 0, "gate": 3},
    {"card_id": "CARD_004", "name": "정가은", "age": 22, "gender": "female", "card_type": "adult",  "count": 0, "gate": 3},
]

TYPE_COLOR = {
    "adult":  (0, 150, 0),
    "senior": (180, 0, 0),
    "teen":   (0, 100, 200),
}

for card in cards:
    card_id  = card["card_id"]
    qr_data  = json.dumps(card, ensure_ascii=False)

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    w, h = qr_img.size

    canvas = Image.new("RGB", (w, h + 70), "white")
    canvas.paste(qr_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font_id   = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", 22)
        font_info = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 16)
    except Exception:
        font_id   = ImageFont.load_default()
        font_info = ImageFont.load_default()

    color     = TYPE_COLOR.get(card["card_type"], (0, 0, 0))
    info_text = f"{card['name']} | {card['age']}세 | {card['gender']} | {card['card_type']}"
    draw.text((w // 2, h + 8),  card_id,   fill=color,      font=font_id,   anchor="mt")
    draw.text((w // 2, h + 36), info_text, fill=(80,80,80), font=font_info, anchor="mt")

    filename = f"{card_id}.png"
    canvas.save(filename)
    print(f" {filename}  ({info_text})")

print("\n총 4개 QR 생성 완료!")
print("스캔 시 JSON 전체가 읽혀 DB 조회 없이 카드 정보를 바로 사용합니다.")
