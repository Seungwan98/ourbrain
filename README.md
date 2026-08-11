# OurBrain Tunnel Crack CV

터널 스캔 이미지에서 균열을 픽셀 단위로 검출하고, 대형 BMP 원본에 대한
균열 유무와 위치를 출력하는 Hugging Face 기반 파인튜닝 프로젝트입니다.

## 프로젝트 문서

프로젝트 현황, 구조, 데이터, v0~v0.3 실험 결과와 운영 절차는
[`docs/README.md`](docs/README.md)에서 확인합니다.

2026-08-02 기준 UPerNet과 SegFormer-B1/B2를 같은 조건으로 비교한 v0.3까지
완료했습니다. 세 후보 모두 validation Dice와 paired group-bootstrap gate에서
v0를 넘지 못해 현재 기준 체크포인트는 v0 epoch 16으로 유지합니다.
정상/hard-negative 검수는 1/200만 완료돼 운영 배포 가능한 최종 모델은 아직 없습니다.

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

# 직접 검수하기 어렵다면 아래 "원격 검수" 절차로 Vercel URL을 공유할 수 있습니다.

# 4. 내려받은 CSV에서 정상으로 확인한 행만 새 manifest에 반영
# review_label에는 negative, normal, no_crack 또는 0을 입력합니다.
uv run ourbrain-cv import-negatives \
  --review ~/Downloads/negative_review_reviewed.csv \
  --manifest artifacts/manifest.csv \
  --output artifacts/manifest_with_negatives.csv

# 5. 학습: train/val/test 각각에 검수된 정상 패치가 없으면 자동 중단
uv run ourbrain-cv training-preflight \
  --config configs/v0_2_a_baseline_with_negatives.yaml \
  --require-local-checkpoint \
  --verify-files

uv run ourbrain-cv train \
  --config configs/v0_2_a_baseline_with_negatives.yaml

# 6. val 그룹에서 운영 임계값 선택 (test는 보정에 사용하지 않음)
uv run ourbrain-cv calibrate \
  --config configs/v0_2_a_baseline_with_negatives.yaml \
  --checkpoint checkpoints/v0.2-a-baseline-with-negatives \
  --minimum-image-recall 0.95 \
  --output artifacts/threshold_calibration.json

# 7. 고정된 임계값으로 보류된 test 그룹을 한 번 평가
uv run ourbrain-cv evaluate \
  --config configs/v0_2_a_baseline_with_negatives.yaml \
  --checkpoint checkpoints/v0.2-a-baseline-with-negatives \
  --calibration artifacts/threshold_calibration.json \
  --split test \
  --output artifacts/test_metrics.json

# 8. test 평가와 같은 고정 임계값으로 대형 BMP 추론
uv run ourbrain-cv infer \
  --config configs/v0_2_a_baseline_with_negatives.yaml \
  --checkpoint checkpoints/v0.2-a-baseline-with-negatives \
  --calibration artifacts/threshold_calibration.json \
  --input '/Volumes/새 볼륨/bmp/0003.bmp' \
  --output outputs/0003
```

`calibrate`는 val의 이미지 단위 민감도가 지정값(기본 95%) 이상인 후보 중
특이도가 가장 높은 임계값을 선택합니다. 동률이면 균열 Dice, boundary F1,
높은 임계값 순으로 결정합니다. 검수된 정상과 균열 이미지가 val에 모두 없으면 본 보정을
중단합니다. 생성 JSON에는 체크포인트·manifest·후처리 설정 출처가 기록되며,
다른 체크포인트나 `minimum_component_pixels` 설정에 잘못 재사용하면
`evaluate`/`infer`가 중단됩니다. test는 임계값 선택에 사용하지 않고, 선택이
끝난 뒤 한 번만 최종 평가합니다. 지정한 민감도를 만족하는 임계값이 하나도
없으면 진단용 곡선은 저장하지만 해당 보정 파일을 평가·추론에 사용하는 것은
자동으로 거부합니다.

평가와 추론은 보정 JSON의 `threshold`, config의
`minimum_component_pixels`, `image_level_minimum_pixels`를 동일하게 적용하므로
보고된 이미지 단위 민감도·특이도는 실제 대형 이미지 판정과 같은 후처리
기준을 사용합니다.

`train`과 `training-preflight`는 반복 실험을 위해 `--model-checkpoint`,
`--output-dir`, `--manifest` override를 지원합니다. `calibrate`, `evaluate`,
`infer`도 `--manifest`를 지원하므로 hard-negative 라운드가 원본 config를
수정하거나 이전 체크포인트를 덮어쓰지 않습니다.

현재 생성된 `data/negative_review/negative_review.csv`에는 전체 원본 86개에서
표본화한 후보 200개와 contact sheet 13장이 준비되어 있습니다. 자동 라벨은
입력하지 않았으며, 담당자가 실제 균열이 없는 패치만 `negative`로 표시해야 합니다.
리뷰 화면은 진행 상황을 브라우저에 저장하고, train/validation/test별 정상
확정 개수를 실시간으로 표시합니다. 균열 의심은 `crack`, 판정 곤란은
`uncertain`으로 두며 두 라벨은 학습 음성으로 반영되지 않습니다.
내보낸 CSV는 Excel/Sheets의 수식 접두사를 중화하며, `import-negatives`가
학습 반영 시 해당 보호 접두사를 다시 안전하게 복원합니다.
200개 모두에 결정이 없으면 import가 중단됩니다. 정상 manifest와 함께 생성되는
`.review.json` 감사 파일의 완료 상태와 SHA-256이 일치하지 않아도 본 학습이
중단되므로, 일부 검수나 사후 변경이 조용히 학습에 반영되지 않습니다.

## Vercel 원격 검수

현재 원격 검수 앱은 다음 주소에 프로덕션 배포되어 있습니다.

```text
https://ourbrain-tunnel-review.vercel.app
```

접근 코드는 저장소에 커밋하지 않으며 Vercel의 `REVIEW_TOKEN` 환경변수와 로컬
`.env.remote-review.local`에만 보관합니다. 검수자 브라우저에서는
`sessionStorage`에만 저장되고 URL, 쿠키, `localStorage`에는 남지 않습니다.

검수자는 다음 기준으로 200장을 판정합니다.

- `N`: 균열 없음 (`negative`) — 학습 정상 패치로 반영
- `C`: 균열 또는 균열 의심 (`crack`) — 정상 패치에서 제외
- `U`: 판단 보류 (`uncertain`) — 정상 패치에서 제외

후보 이미지와 판정은 서울 리전의 private Vercel Blob에 저장합니다. 후보 이미지는
Blob 장애나 사용량 차단 중에도 검수를 계속할 수 있도록 인증된 이미지 함수 번들에도
포함하며, 공개 정적 경로로는 노출하지 않습니다. 접근 코드가 확인된 API만 이미지를
스트리밍하고 응답의 `X-Candidate-Source`가 `bundle` 또는 `blob`으로 실제 소스를
표시합니다. 판정은 immutable 이벤트로 기록하고 동시 저장
충돌은 재판정 대상으로 표시합니다. 모든 200장 판정과 충돌 해소가 끝나기
전에는 CSV export도 차단됩니다.

현재 진행 상황 확인:

```bash
set -a
source .env.remote-review.local
set +a
uv run ourbrain-cv remote-review-status \
  --url https://ourbrain-tunnel-review.vercel.app \
  --summary-only
```

완료된 결과를 내려받아 기존 strict import에 연결:

```bash
uv run ourbrain-cv remote-review-download \
  --url https://ourbrain-tunnel-review.vercel.app \
  --output data/negative_review/negative_review_reviewed.csv

uv run ourbrain-cv import-negatives \
  --review data/negative_review/negative_review_reviewed.csv \
  --manifest artifacts/manifest.csv \
  --output artifacts/manifest_with_negatives.csv
```

후보 데이터나 UI가 바뀌었을 때 재현 가능한 Vercel 번들 생성:

```bash
uv run ourbrain-cv remote-review-bundle \
  --review data/negative_review/negative_review.csv \
  --manifest artifacts/manifest.csv \
  --output build/remote-review-app
```

`build/`와 모든 로컬 환경변수 파일은 Git에서 제외됩니다. 생성 번들의
`private-candidates/`는 비공개 Blob 업로드에만 사용되고 Vercel의 공개 정적
출력에는 포함되지 않습니다. API는 이미지 바이트 SHA-256까지 묶은 dataset 식별자에
속하지 않는 후보 ID를 거부합니다.

## M2 Pro 스모크 테스트

UPerNet의 pyramid pooling은 일부 MPS adaptive-pooling shape를 직접 지원하지
않습니다. 이 프로젝트는 해당 작은 feature map만 CPU로 보내는 호환 계층과,
batch size 1에서 BatchNorm 통계를 고정하는 처리를 포함합니다.

```bash
uv run ourbrain-cv train \
  --config configs/smoke_mps.yaml \
  --device mps \
  --allow-positive-only
uv run ourbrain-cv calibrate \
  --config configs/smoke_mps.yaml \
  --checkpoint checkpoints/smoke-mps \
  --output artifacts/smoke_threshold_calibration.json \
  --max-samples 1 \
  --device mps \
  --allow-positive-only
uv run ourbrain-cv infer \
  --config configs/smoke_mps.yaml \
  --checkpoint checkpoints/smoke-mps \
  --calibration artifacts/smoke_threshold_calibration.json \
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
