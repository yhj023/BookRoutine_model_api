"""
GazeModel 학습 루프.

흐름:
  1) processed_dir(prepare_data.py 결과물)를 학습용/검증용으로 나눠서 불러온다.
  2) 처음 몇 에폭은 backbone을 얼린 채로 head(마지막 레이어)만 학습한다.
  3) --unfreeze_epoch에 도달하면 backbone도 풀어서 더 낮은 학습률로 미세조정한다.
  4) 매 에폭이 끝날 때마다 검증셋 오차(cm)를 계산하고, 가장 좋았던 모델만 저장한다.

실행 예:
    python3 train.py --processed_dir ./processed --epochs 30
"""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, Subset

from data.dataset import GazeEyeDataset
from models.gaze_model import GazeModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")

def group_split(dataset: GazeEyeDataset, val_ratio: float, seed: int = 42):
    """사람(참가자) 단위로 학습/검증을 나눈다."""
    groups = [row.get("group_id") for row in dataset.rows]
    if any(g is None for g in groups):
        print("경고: labels.csv에 group_id가 없어 -> 무작위 분리로 대체함")
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        n_val = int(len(indices) * val_ratio)
        return Subset(dataset, indices[n_val:]), Subset(dataset, indices[:n_val])

    unique_groups = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    n_val_groups = max(1, round(len(unique_groups) * val_ratio))
    val_groups = set(unique_groups[:n_val_groups])
    print(f"전체 {len(unique_groups)}명 -> 검증용으로 사람 {len(val_groups)}명 통째로 분리: {sorted(val_groups)}")

    train_idx = [i for i, g in enumerate(groups) if g not in val_groups]
    val_idx = [i for i, g in enumerate(groups) if g in val_groups]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


@torch.no_grad()
def evaluate(model, loader, device, label_scale) -> float:
    """검증셋에 대한 평균 오차(cm)를 계산. loss(MSE)보다 실제 거리(cm)로 보는 게
    "이 모델이 실사용에서 얼마나 정확한가"를 훨씬 직관적으로 보여준다."""
    model.eval()
    total_error_cm, n = 0.0, 0
    for left, right, label in loader:
        left, right, label = left.to(device), right.to(device), label.to(device)
        pred = model(left, right)
        error_cm = torch.norm((pred - label) * label_scale, dim=1)
        total_error_cm += error_cm.sum().item()
        n += label.size(0)
    return total_error_cm / max(n, 1)


def train(args):
    device = get_device()
    print(f"학습 장치: {device}")

    full_dataset = GazeEyeDataset(args.processed_dir, normalize_label_by=args.label_scale)
    if len(full_dataset) == 0:
        raise ValueError("데이터가 0개야. prepare_data.py가 제대로 실행됐는지 확인해줘.")

    train_set, val_set = group_split(full_dataset, args.val_ratio)
    train_size, val_size = len(train_set), len(val_set)
    if train_size == 0 or val_size == 0:
        raise ValueError(f"학습 {train_size} / 검증 {val_size} — 한쪽이 0개야.")
    print(f"전체 {len(full_dataset)}개 -> 학습 {train_size} / 검증 {val_size}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GazeModel(freeze_backbone=True, pretrained=not args.no_pretrained).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"
    best_val_error = float("inf")

    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch:
            print(f"[{epoch} 에폭] backbone 동결 해제 -> 미세조정 시작 (학습률을 1/10로 낮춤)")
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)

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

        train_loss = running_loss / train_size
        val_error_cm = evaluate(model, val_loader, device, args.label_scale)
        print(f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f}  val_error={val_error_cm:.2f}cm")

        if val_error_cm < best_val_error:
            best_val_error = val_error_cm
            torch.save(
                {"model_state": model.state_dict(), "val_error_cm": val_error_cm, "epoch": epoch},
                ckpt_path,
            )
            print(f"  -> 최고 기록 갱신, {ckpt_path}에 저장함")

    print(f"학습 종료. 최고 검증 오차: {best_val_error:.2f}cm (저장 위치: {ckpt_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./processed")
    parser.add_argument("--out_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_scale", type=float, default=30.0,
                         help="data/dataset.py의 normalize_label_by와 반드시 같은 값이어야 함")
    parser.add_argument("--unfreeze_epoch", type=int, default=15,
                         help="이 에폭부터 backbone 미세조정을 시작함")
    parser.add_argument("--no_pretrained", action="store_true",
                         help="ImageNet 가중치 다운로드가 막힌 환경에서 구조만 테스트할 때 사용")
    args = parser.parse_args()
    train(args)