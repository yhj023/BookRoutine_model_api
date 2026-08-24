"""
개인화 파인튜닝 — 한 사용자의 소량 캘리브레이션 데이터로 마지막 레이어만 재학습.

흐름:
  1) train.py로 학습해둔 범용 모델(best_model.pt)을 불러온다.
  2) 눈 특징을 뽑는 부분(eye_encoder)은 통째로 얼려두고, 맨 마지막
     좌표 예측 레이어(head)만 이 사용자의 캘리브레이션 데이터로 다시 학습한다.
  3) 파인튜닝 전/후 오차를 같이 출력해서, 실제로 개선됐는지 바로 확인할 수 있다.

실행 예:
    python3 finetune_personal.py \
        --base_checkpoint ./checkpoints/best_model.pt \
        --calib_dir ./calib_processed \
        --out_dir ./checkpoints_personal --epochs 20

--calib_dir는 prepare_data.py로 만든 것과 같은 형식(processed_dir)이어야 한다.
"""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import GazeEyeDataset
from models.gaze_model import GazeModel
from train import evaluate, get_device


def split_calib(dataset: GazeEyeDataset, val_ratio: float = 0.2, seed: int = 0):
    """캘리브레이션 데이터가 충분하면(10개 이상) 일부를 떼어 전/후 비교용으로 쓰고,
    너무 적으면 전체를 그대로 써서 "참고용" 수치임을 알린다."""
    n = len(dataset)
    if n < 10:
        print(f"경고: 캘리브레이션 데이터가 {n}개뿐이라 홀드아웃 없이 같은 데이터로 전/후 비교함 (참고용 수치)")
        return dataset, dataset

    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    return Subset(dataset, idx[n_val:]), Subset(dataset, idx[:n_val])


def finetune(args):
    device = get_device()
    print(f"학습 장치: {device}")

    dataset = GazeEyeDataset(args.calib_dir, normalize_label_by=args.label_scale)
    if len(dataset) < 5:
        raise ValueError(f"캘리브레이션 데이터가 너무 적어 ({len(dataset)}개). 최소 10~20장은 필요해.")
    print(f"캘리브레이션 데이터 {len(dataset)}개로 개인화 시작")

    train_set, eval_set = split_calib(dataset)
    train_loader = DataLoader(train_set, batch_size=min(args.batch_size, len(train_set)), shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=min(args.batch_size, len(eval_set)), shuffle=False)

    model = GazeModel(pretrained=False).to(device)
    ckpt = torch.load(args.base_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    base_error = ckpt.get("val_error_cm")
    if base_error is not None:
        print(f"베이스 모델 로드 완료 (범용 모델 검증 오차: {base_error:.2f}cm)")

    # 눈 특징을 뽑는 부분은 통째로 얼리고, 좌표를 예측하는 마지막 head만 재학습
    for p in model.eye_encoder.parameters():
        p.requires_grad = False
    if args.unfreeze_projection:
        for p in model.eye_encoder.project.parameters():
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"이번에 다시 학습할 파라미터: {n_trainable:,} / 전체 {n_total:,}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    before_error = evaluate(model, eval_loader, device, args.label_scale)
    print(f"파인튜닝 전, 이 사용자에 대한 오차: {before_error:.2f}cm")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for left, right, label in train_loader:
            left, right, label = left.to(device), right.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(left, right)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * label.size(0)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"[{epoch:03d}/{args.epochs}] loss={running_loss / len(train_set):.4f}")

    after_error = evaluate(model, eval_loader, device, args.label_scale)
    print(f"파인튜닝 후, 이 사용자에 대한 오차: {after_error:.2f}cm  (개선: {before_error - after_error:+.2f}cm)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "personal_model.pt"
    torch.save({"model_state": model.state_dict(), "personal_error_cm": after_error}, ckpt_path)
    print(f"개인화된 모델 저장 완료: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--calib_dir", required=True, help="이 사용자의 캘리브레이션 눈 crop (prepare_data.py 결과물과 동일 형식)")
    parser.add_argument("--out_dir", default="./checkpoints_personal")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label_scale", type=float, default=30.0,
                         help="base 모델 학습 때 쓴 값과 반드시 같아야 함")
    parser.add_argument("--unfreeze_projection", action="store_true",
                         help="head뿐 아니라 eye_encoder의 마지막 projection layer도 같이 재학습")
    args = parser.parse_args()
    finetune(args)