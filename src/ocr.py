"""
번호판 인식 모듈 (부가 기능)
- 이상탐지된 차량(과속/역주행 등)에 한해 번호판을 인식해 단속 근거로 활용한다.
"""

import re
import cv2
import numpy as np
import easyocr

# 한국 번호판 정규식
#   신형(3자리, 2019년 이후 대부분): 234가2322
#   구형(2자리, 예전 지역명 표기): 12가1234
PLATE_PATTERN = re.compile(r"(\d{2,3})([가-힣])(\d{4})")


def preprocess_plate_image(image):
    """
    OCR 인식률 개선을 위한 전처리.
    번호판 crop이 원본 영상에서 실제로 아주 작은 픽셀 수만 차지하는 경우가 많아서,
    다음 세 단계로 글자를 더 뚜렷하게 만든다:
    1. 업스케일(최소 200px 높이로 확대) — 글자 하나가 인식 가능한 크기가 되도록
    2. 그레이스케일 변환 — 색상 노이즈 제거
    3. CLAHE(대비 제한 적응형 히스토그램 평활화) — 저조도/역광 상황에서 글자 대비를 살림
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image

    # 1. 업스케일 (최소 높이 200px, 원본이 이미 크면 그대로 둠)
    if h < 200:
        scale = 200 / h
        image = cv2.resize(image, (int(w * scale), 200), interpolation=cv2.INTER_CUBIC)

    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 3. CLAHE로 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # EasyOCR은 컬러/그레이 둘 다 받지만, 3채널로 맞춰서 반환 (일관성)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


class PlateOCR:
    def __init__(self, lang_list=None, use_preprocessing=True):
        if lang_list is None:
            lang_list = ["ko", "en"]
        self.reader = easyocr.Reader(lang_list, gpu=False)
        self.use_preprocessing = use_preprocessing

    def read(self, plate_image) -> str | None:
        """
        번호판 이미지에서 텍스트를 추출한다.
        한국 번호판 형식(숫자2~3 + 한글1 + 숫자4)에 맞는 부분만 추출해서 반환하고,
        형식에 안 맞으면 None을 반환한다 (오인식 필터링).
        """
        result = self.read_debug(plate_image)
        return result["parsed"]

    def read_debug(self, plate_image) -> dict:
        """
        디버깅용: EasyOCR이 실제로 뭐라고 읽었는지(raw_text)와,
        정규식 필터링을 거친 최종 결과(parsed)를 함께 반환한다.
        원인 분석(왜 인식이 안 됐는지)이 필요할 때 사용한다.
        """
        target = plate_image
        if self.use_preprocessing:
            # 파일 경로(str)로 들어오면 먼저 이미지로 읽어서 전처리
            if isinstance(plate_image, str):
                loaded = cv2.imread(plate_image)
                if loaded is not None:
                    target = preprocess_plate_image(loaded)
            else:
                target = preprocess_plate_image(plate_image)

        results = self.reader.readtext(target)
        raw_text = "".join([res[1] for res in results])
        confidences = [res[2] for res in results]  # 각 글자 조각의 인식 신뢰도
        parsed = self._postprocess(raw_text)
        return {"raw_text": raw_text, "confidences": confidences, "parsed": parsed}

    def _postprocess(self, text: str) -> str | None:
        # 1차: 숫자/한글 이외 문자(공백, 특수문자) 제거
        cleaned = re.sub(r"[^0-9가-힣]", "", text)

        # 2차: 번호판 정규식에 맞는 부분만 추출
        match = PLATE_PATTERN.search(cleaned)
        if match:
            digits1, hangul, digits2 = match.groups()
            return f"{digits1}{hangul}{digits2}"

        # 형식에 안 맞으면 오인식으로 간주하고 None 반환
        return None


