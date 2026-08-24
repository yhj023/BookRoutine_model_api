"""
원본 데이터셋을 순회하면서 EyeExtractor로 눈 crop을 만들어 processed_dir에
저장하고, labels.csv를 생성한다.

사용 예:
    python3 prepare_data.py \
        --raw_dir /path/to/images \
        --raw_labels /path/to/raw_labels.csv \
        --out_dir ./processed

raw_labels.csv 기대 형식: image_path,gaze_x_cm,gaze_y_cm
"""

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from data.eye_extractor import EyeExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="원본 이미지들이 있는 폴더")
    parser.add_argument("--raw_labels", required=True, help="image_path,gaze_x_cm,gaze_y_cm 형식의 csv")
    parser.add_argument("--out_dir", default="./processed")
    parser.add_argument("--eye_size", type=int, default=64)
    parser.add_argument("--face_size", type=int, default=224)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "left").mkdir(parents=True, exist_ok=True)
    (out_dir / "right").mkdir(parents=True, exist_ok=True)

    with open(args.raw_labels, newline="") as f:
        raw_rows = list(csv.DictReader(f))

    kept, not_found, no_face = 0, 0, 0
    sample_missing_paths = []  # 원인 진단용: 못 찾은 경로를 몇 개만 기억해둔다

    with EyeExtractor(face_size=args.face_size, eye_size=args.eye_size) as extractor, \
         open(out_dir / "labels.csv", "w", newline="") as out_f:

        writer = csv.writer(out_f)
        writer.writerow(["sample_id", "left_eye_path", "right_eye_path", "gaze_x_cm", "gaze_y_cm", "group_id"])

        for i, row in enumerate(tqdm(raw_rows, desc="눈 crop 추출 중")):
            full_path = raw_dir / row["image_path"]
            img = cv2.imread(str(full_path))
            if img is None:
                not_found += 1
                if len(sample_missing_paths) < 5:
                    sample_missing_paths.append(str(full_path))
                continue

            sample = extractor.process(img)
            if sample is None:
                no_face += 1
                continue

            left_path = f"left/{i:07d}.png"
            right_path = f"right/{i:07d}.png"
            cv2.imwrite(str(out_dir / left_path), sample.left_eye)
            cv2.imwrite(str(out_dir / right_path), sample.right_eye)

            # image_path의 맨 앞 폴더 이름 = 이 사진이 누구 것인지 (예: "p00/day01/0005.jpg" -> "p00")
            group_id = row["image_path"].split("/")[0]
            writer.writerow([i, left_path, right_path, row["gaze_x_cm"], row["gaze_y_cm"], group_id])
            kept += 1

    print(f"완료: {kept}개 저장 / 파일 못 찾음 {not_found}개 / 얼굴 미검출 {no_face}개")
    if sample_missing_paths:
        print("\n못 찾은 경로 예시 (이 경로들이 실제로 존재하는지 확인해봐):")
        for p in sample_missing_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()