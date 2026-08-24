"""
전이학습 기반 시선추적 모델.

- EyeEncoder: ImageNet으로 사전학습된 MobileNetV3-Small을 특징 추출기로 사용.
  왼쪽/오른쪽 눈에 "같은" 인코더를 두 번 재사용한다 (가중치 공유) —
  두 눈은 생김새 원리가 같으므로 따로 학습시킬 필요가 없다.
- GazeModel: 두 눈의 특징을 합쳐(concat) 화면 좌표 (x, y)를 회귀 예측.

이 파일은 "모델 구조 정의"까지다. 실제 학습 루프(옵티마이저, 에폭 반복,
저장/로깅)는 다음 단계인 train.py에서 다룬다.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class EyeEncoder(nn.Module):
    """단일 눈 이미지 -> 특징 벡터. 전이학습의 시작점(ImageNet 사전학습)을 사용한다."""

    def __init__(self, out_dim: int = 256, freeze_backbone: bool = True, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(576, out_dim)  # MobileNetV3-Small 마지막 채널 수 = 576

        if freeze_backbone:
            # 1차 학습에서는 backbone을 얼리고 head만 학습 -> 이후 필요하면 unfreeze해서 미세조정
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pool(self.features(x)).flatten(1)
        return self.project(feat)


class GazeModel(nn.Module):
    """왼쪽눈 + 오른쪽눈 -> 화면 좌표 (x, y) 예측."""

    def __init__(self, eye_feat_dim: int = 256, freeze_backbone: bool = True, pretrained: bool = True):
        super().__init__()
        self.eye_encoder = EyeEncoder(out_dim=eye_feat_dim, freeze_backbone=freeze_backbone, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Linear(eye_feat_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # 출력: 정규화된 화면 좌표 (x, y)
        )

    def forward(self, left_eye: torch.Tensor, right_eye: torch.Tensor) -> torch.Tensor:
        left_feat = self.eye_encoder(left_eye)
        right_feat = self.eye_encoder(right_eye)  # 같은 인코더를 재사용 (가중치 공유)
        combined = torch.cat([left_feat, right_feat], dim=1)
        return self.head(combined)

    def unfreeze_backbone(self):
        """개인화 파인튜닝 단계에서 마지막 레이어 외에 backbone 일부도 살짝 풀고 싶을 때 사용."""
        for p in self.eye_encoder.features.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    # 구조가 제대로 조립되는지 확인하는 스모크 테스트 (실제 학습 데이터 없이도 실행 가능)
    # pretrained=False: 이 환경은 외부 가중치 서버(download.pytorch.org) 접근이 막혀 있어
    # 구조 검증용으로만 무작위 초기화 사용. 실제 학습 환경에서는 pretrained=True로 실행할 것.
    import sys
    pretrained = "--no-pretrained" not in sys.argv
    model = GazeModel(pretrained=pretrained)
    dummy_left = torch.randn(4, 3, 64, 64)
    dummy_right = torch.randn(4, 3, 64, 64)
    out = model(dummy_left, dummy_right)
    print("출력 shape:", out.shape)  # 기대값: (4, 2)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터: {n_params:,} / 학습 대상: {n_trainable:,}")
