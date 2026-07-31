# 데이터와 라벨링

## 원본 위치와 정책

주 데이터 위치:

```text
/Volumes/새 볼륨
```

원본은 읽기 전용으로 사용합니다. manifest, 검수 결과, 변환 산출물과 체크포인트는
저장소 또는 Windows 학습 디렉터리에 별도로 기록합니다.

## 감사 결과

2026-07-29 기준:

| 항목 | 수량 |
|---|---:|
| 패치 이미지 | 1,886 |
| 라벨 파일 | 1,237 |
| 이름이 매칭된 페어 | 1,224 |
| 정상 디코딩 페어 | 1,223 |
| 이미지에 라벨 없음 | 662 |
| 이미지 없는 라벨 | 13 |
| 유효 원본 그룹 | 89 |
| 제외된 손상 페어 | 1 |

상세 파일 목록과 손상 파일 정보는 [데이터 감사 결과](DATA_AUDIT.md)에 있습니다.

## 마스크의 의미

마스크가 있다는 것은 해당 이미지에 픽셀 단위 라벨이 있다는 뜻입니다.

- 원본 마스크의 **검정 픽셀**을 `crack=1`로 변환합니다.
- 나머지 픽셀은 `background=0`입니다.
- 크기가 다른 마스크는 category를 보존하기 위해 nearest-neighbor로 512×512에
  맞춥니다.
- 원본 크기는 512×512 마스크 589장, 682×682 마스크 633장, 711×711 마스크
  1장이며 634장은 loader에서 이미지 크기에 맞춰집니다.
- crack 양성 픽셀은 패치당 51~2,133개입니다.
- 총 crack 픽셀은 710,984개입니다.

2026-07-31 preflight에서 유효 manifest의 이미지 1,223장과 마스크 1,223장,
총 2,071,077,098 bytes를 실제로 모두 디코딩했습니다.

현재 마스크는 실제 균열 폭 영역보다 중심선에 가까운 얇은 라벨입니다. 따라서
균열의 존재와 위치 학습에는 사용할 수 있지만, 균열 폭을 물리 단위로 계측하는
정답으로 사용하면 안 됩니다.

## 라벨 없는 662장의 해석

라벨 파일이 없다는 사실만으로 `균열 없음`을 의미하지 않습니다. 다음 가능성이
모두 존재합니다.

- 실제 정상 이미지
- 균열이 있으나 아직 라벨링되지 않은 이미지
- 파일명 불일치
- 라벨 누락

따라서 이 이미지를 자동으로 정상 데이터에 넣지 않습니다. 사람이 직접 확인한
패치만 정상 학습 샘플로 승격합니다.

## 정상/hard-negative 검수

원본 BMP 86개에서 기존 균열 패치와 겹치지 않도록 512×512 후보 200장을
표본화했습니다.

| split | 후보 |
|---|---:|
| train | 136 |
| validation | 38 |
| test | 26 |
| 합계 | 200 |

2026-07-31 조회 상태:

| 상태 | 수량 |
|---|---:|
| 검수 완료 | 1 |
| 미검수 | **199** |
| `negative` | 0 |
| `crack` | 0 |
| `uncertain` | 1 |
| 충돌 | 0 |

검수 URL:

```text
https://ourbrain-tunnel-review.vercel.app
```

검수자는 다음 기준을 사용합니다.

- `N` / `negative`: 균열 없음 — 정상 학습 데이터로 사용
- `C` / `crack`: 균열 또는 균열 의심 — 정상 데이터에서 제외
- `U` / `uncertain`: 판단 곤란 — 정상 데이터에서 제외

이음부, 볼트, 케이블, 타일 패턴, 오염, 누수 흔적, 조명 반사처럼 균열과 혼동하기
쉬운 정상 구조물을 의도적으로 `negative`로 확보해야 실제 오탐률을 줄일 수 있습니다.

## 검수 결과 반영 게이트

200장 전체가 결정되지 않으면 CSV export와 strict import가 중단됩니다.
`import-negatives`는 다음 조건을 확인합니다.

1. 모든 후보에 결정이 존재함
2. 충돌이 모두 해소됨
3. 검수 감사 파일이 완료 상태임
4. manifest와 감사 파일의 SHA-256이 일치함
5. 누적된 모든 검수 정상 PNG의 실제 바이트 SHA-256이 import 시점과 일치함
6. `negative`로 확정된 행만 추가됨

```bash
set -a
source .env.remote-review.local
set +a

uv run ourbrain-cv remote-review-status \
  --url https://ourbrain-tunnel-review.vercel.app \
  --summary-only

uv run ourbrain-cv remote-review-download \
  --url https://ourbrain-tunnel-review.vercel.app \
  --output data/negative_review/negative_review_reviewed.csv

uv run ourbrain-cv import-negatives \
  --review data/negative_review/negative_review_reviewed.csv \
  --manifest artifacts/manifest.csv \
  --output artifacts/manifest_with_negatives.csv
```

## 현재 데이터로 가능한 것과 불가능한 것

가능:

- 양성 패치의 균열 위치를 학습하는 개발 실험
- 모델 코드, GPU/MPS 실행 경로와 대형 이미지 추론 검증
- 양성 validation에서 pixel Dice/IoU/recall 비교

불가능:

- 실제 정상 이미지에 대한 오탐률 증명
- 신뢰할 수 있는 이미지 단위 specificity 산출
- 운영 임계값 확정
- 최종 모델 승인

양성 validation만으로 출력되는 이미지 단위 specificity `1.0`은 정상 표본이 0개라
생기는 빈 분모 기본값이며, 성능 증거가 아닙니다.
