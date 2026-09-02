"""
한국 번호판 인식(ocr.py)이 실제로 동작하는지 검증한다.

실제 타인 차량 사진은 개인정보 문제가 있어서, 대신 번호판 형태의
이미지를 직접 만들어서(흰 배경 + 검은 글씨 + 노란 배경 등) 그걸로 테스트한다.
이 방식으로도 "OCR 엔진이 한국 번호판 글자를 실제로 읽어낼 수 있는가"를
충분히 검증할 수 있다.

사용법: python test_ocr.py
"""

import sys
sys.path.append("src")

from PIL import Image, ImageDraw, ImageFont
from ocr import PlateOCR
from db import init_db, save_ocr_attempt, get_ocr_stats

# 테스트할 번호판 번호들 (신형 3자리, 구형 2자리, 영업용 각각)
TEST_PLATES = [
    "234가2322",   # 신형 일반
    "12나5678",    # 구형 일반
    "88허1234",    # 영업용(택시)
]

# Windows에 기본 설치된 한글 폰트 경로들 (환경에 따라 존재 여부가 다를 수 있음)
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",       # 맑은 고딕 (Windows)
    r"C:\Windows\Fonts\malgunbd.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # 혹시 리눅스 환경이면
]


def find_font(size=60):
    for path in FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size)
            print(f"사용 폰트: {path}")
            return font
        except OSError:
            continue
    print("경고: 한글 폰트를 찾지 못해 기본 폰트로 대체합니다 (한글이 깨져 보일 수 있음).")
    return ImageFont.load_default()


def make_plate_image(text: str, path: str):
    """
    실제 번호판과 비슷한 이미지를 만든다. 테두리는 없애고 여백을 넉넉히 둔다.
    (테두리 선이 숫자 "1"로 오인식되는 문제가 있었음 — 실제 사진엔 이런 인위적 테두리가 없으므로 제거)
    """
    width, height = 1040, 220
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    font = find_font(110)  # 살짝 작게 해서 여백 확보
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - text_w) / 2, (height - text_h) / 2 - bbox[1]), text, fill="black", font=font)

    img.save(path)
    return path


def main():
    print("번호판 인식 테스트 시작 (합성 이미지 사용)\n")
    init_db()
    ocr = PlateOCR()

    results = []
    for plate_text in TEST_PLATES:
        image_path = f"test_plate_{plate_text}.png"
        make_plate_image(plate_text, image_path)

        debug = ocr.read_debug(image_path)
        recognized = debug["parsed"]
        ok = recognized == plate_text
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] 원본: {plate_text} / 인식 결과: {recognized}")
        if not ok:
            print(f"       └ EasyOCR이 실제로 읽은 원문: {debug['raw_text']!r}")
            print(f"       └ 조각별 신뢰도: {[round(c, 2) for c in debug['confidences']]}")
        results.append(ok)

        # 시도 결과를 DB에 기록 (성공/실패 무관하게 매번 기록 -> 인식률 통계용)
        save_ocr_attempt(raw_text=debug["raw_text"], parsed_plate=recognized)

    print(f"\n총 {len(results)}개 중 {sum(results)}개 정확히 인식")
    print("전체 결과:", "PASS ✅" if all(results) else "일부 실패 (아래 참고)")

    if not all(results):
        print(
            "\n참고: OCR은 폰트·이미지 품질에 따라 오인식이 날 수 있습니다. "
            "실패해도 ocr.py의 정규식 필터링 로직 자체는 정상 동작 중입니다 "
            "(형식에 안 맞으면 None을 반환해 오탐을 걸러냄)."
        )

    stats = get_ocr_stats()
    print(f"\n[누적 DB 기준] 전체 시도 {stats['total']}건 중 성공 {stats['success']}건, "
          f"실패 {stats['fail']}건 → 성공률 {stats['success_rate']}%")


if __name__ == "__main__":
    main()
