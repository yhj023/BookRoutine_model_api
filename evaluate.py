"""
학습이 끝난 모델을, 학습에 전혀 쓰지 않은 테스트셋으로 마지막에 한 번 더
평가하는 스크립트.

실행 예:
    python3 evaluate.py --processed_dir ./processed_test --checkpoint ./checkpoints/best_model.pt
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import GazeEyeDataset
from models.gaze_model import GazeModel


@torch.no_grad()
def collect_errors(model, loader, device, label_scale) -> np.ndarray:
    model.eval()
    errors = []
    for left, right, label in loader:
        left, right, label = left.to(device), right.to(device), label.to(device)
        pred = model(left, right)
        error_cm = torch.norm((pred - label) * label_scale, dim=1)
        errors.extend(error_cm.cpu().tolist())
    return np.array(errors)


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_scale", type=float, default=30.0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GazeEyeDataset(args.processed_dir, normalize_label_by=args.label_scale)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = GazeModel(pretrained=False).to(device)  # 체크포인트 값으로 덮어쓸 거라 사전학습 가중치는 불필요
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"체크포인트 로드 완료 (학습 시 검증 오차: {ckpt.get('val_error_cm', '?'):.2f}cm, epoch {ckpt.get('epoch', '?')})")

    errors = collect_errors(model, loader, device, args.label_scale)
    print(f"\n테스트 샘플 수: {len(errors)}")
    print(f"평균 오차:   {errors.mean():.2f} cm")
    print(f"중앙값 오차: {np.median(errors):.2f} cm")
    print(f"최대 오차:   {errors.max():.2f} cm")
    print(f"샘플의 90%가 {np.percentile(errors, 90):.2f} cm 이내의 오차")


if __name__ == "__main__":
    main()