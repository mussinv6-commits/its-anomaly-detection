"""
번호판 인식 모듈 (부가 기능)
- 이상탐지된 차량(과속/역주행 등)에 한해 번호판을 인식해 단속 근거로 활용한다.
"""

import re
import easyocr

# 한국 번호판 정규식
#   신형(3자리, 2019년 이후 대부분): 234가2322
#   구형(2자리, 예전 지역명 표기): 12가1234
PLATE_PATTERN = re.compile(r"(\d{2,3})([가-힣])(\d{4})")


class PlateOCR:
    def __init__(self, lang_list=None):
        if lang_list is None:
            lang_list = ["ko", "en"]
        self.reader = easyocr.Reader(lang_list, gpu=False)

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
        results = self.reader.readtext(plate_image)
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

