"""
PaddleOCR 기반 번호판 인식 모듈.

ocr.py(EasyOCR 버전)와 정확히 같은 인터페이스(read, read_debug)를 제공해서,
compare_ocr_engines.py에서 두 엔진을 나란히 비교할 수 있게 한다.

PaddleOCR은 중국어/한자권 문자 인식에서 EasyOCR보다 강하다고 알려져 있어,
번호판의 한글 인식률이 여기서 개선되는지 확인하기 위해 도입한다.

설치(무거운 편이라 시간이 좀 걸릴 수 있음):
    pip install paddlepaddle paddleocr --break-system-packages
"""

import re

PLATE_PATTERN = re.compile(r"(\d{2,3})([가-힣])(\d{4})")


class PlateOCRPaddle:
    def __init__(self):
        from paddleocr import PaddleOCR
        # lang="korean": 한국어 인식 모델. use_angle_cls: 기울어진 텍스트 보정.
        # show_log는 PaddleOCR 버전에 따라 지원하지 않는 경우가 있어 제거 (버전별 API 차이 대응)
        self.reader = PaddleOCR(use_angle_cls=True, lang="korean")

    def _postprocess(self, text: str):
        cleaned = re.sub(r"[^0-9가-힣]", "", text)
        match = PLATE_PATTERN.search(cleaned)
        if match:
            d1, h, d2 = match.groups()
            return f"{d1}{h}{d2}"
        return None

    def read(self, plate_image) -> str | None:
        return self.read_debug(plate_image)["parsed"]

    def read_debug(self, plate_image) -> dict:
        # PaddleOCR 최신 버전(v5+)은 ocr() 호출 시 cls 인자를 받지 않음 (use_angle_cls 생성자 옵션으로 이미 처리됨)
        try:
            result = self.reader.ocr(plate_image, cls=True)
        except TypeError:
            result = self.reader.ocr(plate_image)

        texts, confidences = [], []
        if result and result[0] is not None:
            first = result[0]
            if isinstance(first, dict) and "rec_texts" in first:
                # PaddleOCR v5(PaddleX 파이프라인) 결과 형식: dict에 rec_texts/rec_scores 리스트로 담김
                texts = list(first.get("rec_texts", []))
                confidences = list(first.get("rec_scores", []))
            else:
                # 이전 버전 결과 형식: [[box, (text, confidence)], ...]
                for line in first:
                    try:
                        text, conf = line[1]
                        texts.append(text)
                        confidences.append(conf)
                    except (TypeError, ValueError, IndexError):
                        continue

        raw_text = "".join(texts)
        parsed = self._postprocess(raw_text)
        return {"raw_text": raw_text, "confidences": confidences, "parsed": parsed}
