"""
영상의 첫 프레임을 이미지로 저장해서, 화면 속 기준점 4개의 픽셀 좌표를 확인하고
그에 대응하는 실제 지면 좌표(미터)를 입력해서 호모그래피 행렬을 계산한다.

사용법:
    1단계 - 기준 프레임 이미지 뽑기:
        python calibrate_homography.py --video 5.mp4 --extract-frame

        -> calibration_frame.jpg가 생성됨. 이 이미지를 열어서, 화면 속에서
           "실제 거리를 알 수 있는 사각형 모양의 기준점 4개"의 픽셀 좌표를 확인한다.
           (예: 횡단보도 네 모서리, 차선 폭이 일정한 구간의 네 점 등)

    2단계-A (호모그래피, 가로/세로 다 보정, 4점 다 정확해야 함) - 좌표 입력해서 계산:
        python calibrate_homography.py \\
            --image-points 1200,1500 1600,1500 1600,1300 1200,1300 \\
            --real-points 0,0 3.5,0 3.5,10 0,10 \\
            --output models/homography_5mp4.npy

        --image-points: 화면 속 4개 점의 픽셀 좌표 (x,y 순서, 콤마로 구분, 공백으로 점 구분)
        --real-points: 그 4개 점의 실제 지면 좌표 (미터, 임의 원점 기준)
                       예시는 "폭 3.5m(표준 차선 폭) x 길이 10m" 직사각형을 가정한 것

    2단계-B (깊이 스케일, 세로 위치만 보정, 더 단순하고 가로 추측이 없어 안전함) - 권장:
        python calibrate_homography.py \\
            --depth-points 1580,0.00382 1400,0.00897 \\
            --output models/depthscale_5mp4.npy \\
            --mode depth

        --depth-points: (화면 y좌표, 그 지점에서 검증된 meters_per_pixel) 쌍을 여러 개
                        예: 표준 규격 차량(버스 폭 2.495m, 승용차 폭 약 1.8m)을 화면에서
                        실제로 찾아 픽셀폭을 재고 "실제폭/픽셀폭"으로 계산한 값을 사용

    3단계 - main.py에서 사용:
        python main.py --video ..\\5.mp4 --homography models/homography_5mp4.npy
        (또는 depth 방식: --homography models/depthscale_5mp4.npy --homography-mode depth)
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.append("src")

from perspective import PerspectiveCalibrator, DepthScaleCalibrator


def extract_first_frame(video_path: str, output_path: str = "calibration_frame.jpg"):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"{video_path}에서 프레임을 읽지 못했습니다.")
        return
    cv2.imwrite(output_path, frame)
    h, w = frame.shape[:2]
    print(f"{output_path} 저장 완료 (해상도: {w}x{h})")
    print("이 이미지를 열어서 기준점 4개의 픽셀 좌표를 확인한 뒤, --image-points로 입력하세요.")


def parse_points(point_strs):
    points = []
    for p in point_strs:
        x, y = p.split(",")
        points.append((float(x), float(y)))
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="기준 프레임을 뽑을 영상 경로")
    parser.add_argument("--extract-frame", action="store_true", help="영상의 첫 프레임을 이미지로 저장")
    parser.add_argument("--mode", choices=["homography", "depth"], default="homography", help="보정 방식 선택")
    parser.add_argument("--image-points", nargs=4, help="[homography 모드] 화면 속 4개 점의 픽셀 좌표 (x,y 형식, 4개)")
    parser.add_argument("--real-points", nargs=4, help="[homography 모드] 대응하는 실제 지면 좌표 (미터, x,y 형식, 4개)")
    parser.add_argument(
        "--depth-points", nargs="+",
        help="[depth 모드] '화면y,meters_per_pixel' 쌍을 2개 이상. 예: 1580,0.00382 1400,0.00897",
    )
    parser.add_argument("--output", default="models/homography.npy", help="계산된 보정값 저장 경로")
    args = parser.parse_args()

    if args.extract_frame:
        if not args.video:
            print("--video로 영상 경로를 지정해주세요.")
            return
        extract_first_frame(args.video)
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "depth":
        if not args.depth_points or len(args.depth_points) < 2:
            print("--depth-points로 (화면y,meters_per_pixel) 쌍을 2개 이상 입력해주세요.")
            print("예: --depth-points 1580,0.00382 1400,0.00897")
            return
        reference_points = []
        for p in args.depth_points:
            y, scale = p.split(",")
            reference_points.append((float(y), float(scale)))
        calibrator = DepthScaleCalibrator(reference_points)

        import pickle
        with open(args.output, "wb") as f:
            pickle.dump(calibrator, f)
        print(f"깊이보정(DepthScaleCalibrator) 저장 완료: {args.output}")
        print("\n검증 (입력한 지점에서 그대로 스케일이 나와야 함):")
        for y, expected_scale in reference_points:
            actual = calibrator.scale_at(y)
            print(f"  y={y}: {actual:.5f} (기대값 {expected_scale:.5f})")
        return

    if not args.image_points or not args.real_points:
        print("--image-points와 --real-points를 각각 4개씩 입력해주세요.")
        print("사용법은 이 파일 상단의 설명을 참고하세요.")
        return

    image_points = parse_points(args.image_points)
    real_points = parse_points(args.real_points)

    calibrator = PerspectiveCalibrator.from_points(image_points, real_points)
    calibrator.save(args.output)
    print(f"호모그래피 저장 완료: {args.output}")

    # 검증: 기준점끼리의 거리가 입력한 실제 거리와 맞게 나오는지 확인
    print("\n검증 (기준점 간 거리가 입력한 실제 좌표와 일치해야 함):")
    for i in range(4):
        j = (i + 1) % 4
        calculated = calibrator.distance_meters(image_points[i], image_points[j])
        expected = ((real_points[i][0] - real_points[j][0]) ** 2 + (real_points[i][1] - real_points[j][1]) ** 2) ** 0.5
        print(f"  점{i+1}-점{j+1}: 계산값 {calculated:.2f}m (기대값 {expected:.2f}m)")


if __name__ == "__main__":
    main()
