"""
번호판 인식 모듈 (부가 기능)
- 이상탐지된 차량(과속/역주행 등)에 한해 번호판을 인식해 단속 근거로 활용한다.
"""

import re
import easyocr


class PlateOCR:
    def __init__(self, lang_list=None):
        if lang_list is None:
            lang_list = ["ko", "en"]
        self.reader = easyocr.Reader(lang_list, gpu=False)

    def read(self, plate_image) -> str:
        results = self.reader.readtext(plate_image)
        text = "".join([res[1] for res in results])
        return self._postprocess(text)

    def _postprocess(self, text: str) -> str:
        cleaned = re.sub(r"[^0-9가-힣]", "", text)
        return cleaned
