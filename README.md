# OurBrain Tunnel Crack CV

터널 스캔 이미지에서 균열을 픽셀 단위로 검출하고, 대형 BMP 원본에 대한
균열 유무와 위치를 출력하는 Hugging Face 기반 파인튜닝 프로젝트입니다.

## 안전 원칙

- `/Volumes/새 볼륨`의 원본은 읽기 전용 입력으로만 사용합니다.
- manifest, 정규화 결과, 체크포인트와 추론 결과는 이 저장소 아래에만 기록합니다.
- 정상 후보는 자동으로 정상 라벨이 되지 않습니다. `review.csv`에서 사람이 검토한
  항목만 학습 데이터에 추가해야 합니다.

## 기준 모델

- 세그멘테이션: `openmmlab/upernet-swin-tiny` (MIT)
- 입력 패치: 512×512
- 출력 클래스: `background=0`, `crack=1`
- 손실 함수: focal + Dice + boundary
- 분할 정책: 동일 원본 번호가 여러 split에 섞이지 않는 group split

## 설치

```bash
uv sync --extra dev
```

명령 목록은 다음과 같이 확인합니다.

```bash
uv run ourbrain-cv --help
```

## 기본 실행 흐름

```bash
# 1. 원본을 수정하지 않고 데이터 감사 및 manifest 생성
uv run ourbrain-cv prepare \
  --data-root '/Volumes/새 볼륨/train' \
  --manifest artifacts/manifest.csv \
  --audit artifacts/data_audit.json

# 2. 정상/hard-negative 후보 생성 (아직 학습 라벨이 아님)
uv run ourbrain-cv negative-candidates \
  --raw-root '/Volumes/새 볼륨/bmp' \
  --manifest artifacts/manifest.csv \
  --output data/negative_review \
  --max-candidates 200

# 3. 브라우저에서 후보를 한 장씩 검수하고 CSV 내보내기
uv run ourbrain-cv review-ui \
  --review data/negative_review/negative_review.csv \
  --manifest artifacts/manifest.csv \
  --output data/negative_review/review.html \
  --serve

# 브라우저에서 N/C/U로 검수하고 "검수 CSV 내보내기"를 누른 뒤
# 이 터미널에서 Ctrl-C로 리뷰 서버를 종료합니다.

# 4. 내려받은 CSV에서 정상으로 확인한 행만 새 manifest에 반영
# review_label에는 negative, normal, no_crack 또는 0을 입력합니다.
uv run ourbrain-cv import-negatives \
  --review ~/Downloads/negative_review_reviewed.csv \
  --manifest artifacts/manifest.csv \
  --output artifacts/manifest_with_negatives.csv

# 5. 학습: train/val/test 각각에 검수된 정상 패치가 없으면 자동 중단
uv run ourbrain-cv train --config configs/upernet_swin_tiny.yaml

# 6. 보류된 test 그룹 평가
uv run ourbrain-cv evaluate \
  --config configs/upernet_swin_tiny.yaml \
  --checkpoint checkpoints/upernet-swin-tiny \
  --split test \
  --output artifacts/test_metrics.json

# 7. 대형 BMP 추론
uv run ourbrain-cv infer \
  --config configs/upernet_swin_tiny.yaml \
  --checkpoint checkpoints/upernet-swin-tiny \
  --input '/Volumes/새 볼륨/bmp/0003.bmp' \
  --output outputs/0003
```

현재 생성된 `data/negative_review/negative_review.csv`에는 전체 원본 86개에서
표본화한 후보 200개와 contact sheet 13장이 준비되어 있습니다. 자동 라벨은
입력하지 않았으며, 담당자가 실제 균열이 없는 패치만 `negative`로 표시해야 합니다.
리뷰 화면은 진행 상황을 브라우저에 저장하고, train/validation/test별 정상
확정 개수를 실시간으로 표시합니다. 균열 의심은 `crack`, 판정 곤란은
`uncertain`으로 두며 두 라벨은 학습 음성으로 반영되지 않습니다.
내보낸 CSV는 Excel/Sheets의 수식 접두사를 중화하며, `import-negatives`가
학습 반영 시 해당 보호 접두사를 다시 안전하게 복원합니다.

## M2 Pro 스모크 테스트

UPerNet의 pyramid pooling은 일부 MPS adaptive-pooling shape를 직접 지원하지
않습니다. 이 프로젝트는 해당 작은 feature map만 CPU로 보내는 호환 계층과,
batch size 1에서 BatchNorm 통계를 고정하는 처리를 포함합니다.

```bash
uv run ourbrain-cv train \
  --config configs/smoke_mps.yaml \
  --device mps \
  --allow-positive-only
uv run ourbrain-cv infer \
  --config configs/smoke_mps.yaml \
  --checkpoint checkpoints/smoke-mps \
  --input '/Volumes/새 볼륨/train/crack/0005_022_001.png' \
  --output outputs/smoke-0005 \
  --device mps
```

`smoke-mps`는 실행 경로 검증용으로 두 샘플만 학습하므로 실제 판정에 사용하면
안 됩니다.

대형 원본에서 임계값 초과 픽셀이 `maximum_positive_ratio`(기본 25%)를 넘으면
연결요소 분석과 균열 유무 결정을 보류합니다. 이는 보정되지 않은 모델이 과도한
양성을 출력할 때 메모리와 잘못된 자동 판정을 함께 방지하는 품질 게이트입니다.

## 검증

```bash
uv run pytest
uv run ruff check .
```

## 데이터 주의사항

현재 확인된 데이터는 균열 중심선 마스크가 있는 양성 패치 위주입니다. 따라서
픽셀 단위 세그멘테이션 학습은 가능하지만, 이미지 단위의 `균열 없음` 판정과
오탐률을 신뢰하려면 이음부·볼트·오염·케이블을 포함한 정상/hard-negative
패치를 별도로 검수해 추가해야 합니다.

실제 데이터 감사 결과는 [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md)를 참조하세요.
