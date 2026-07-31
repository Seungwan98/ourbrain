# 우선순위 실행 계획

작성일: 2026-07-31

## 목표

현재 양성 데이터 중심의 개발 모델을, 검수된 정상 데이터와 held-out test 근거를
갖춘 터널 균열 판정 모델로 발전시킵니다.

최종 성공 조건:

1. 사람 검수 정상 데이터가 train/validation/test에 모두 존재
2. 원본 group 누수 0
3. validation 이미지 recall 0.95 이상
4. 해당 recall 조건에서 specificity가 가장 높은 threshold 선택
5. 고정 threshold로 held-out test 평가 완료
6. 대형 BMP 파일럿과 현장 오류 검토 완료

specificity의 최종 합격 수치는 OurBrain 및 현장 담당자와 별도로 확정합니다.

## 현재 기준점

| 항목 | 상태 |
|---|---|
| v0 best | epoch 16, validation crack Dice 0.256632 |
| v0.1 best | epoch 1, validation crack Dice 0.254753 |
| 동일 조건 재평가 | v0 Dice 0.257609, v0.1 Dice 0.253832, validation 221장 |
| v0.2-dev 사전 결과 | dev A Dice 0.258174, dev B Dice 0.254591, test 미개봉 |
| 대형 BMP smoke benchmark | 17.98MP, 모델당 약 21.15초, 0.85MP/초 |
| 현재 기준 모델 | v0 epoch 16 |
| 정상 후보 검수 | 1/200 완료(uncertain 1, negative 0), 199장 남음 |
| 운영 배포 가능 여부 | 불가 |

## 2026-07-31 실행 준비 상태

| 단계 | 실제 데이터 작업 | 자동화/검증 준비 |
|---|---|---|
| P0 | 체크포인트·해시 보존 완료, Tailscale 마무리는 사용자 요청으로 연기 | LAN headless SSH·AC 덮개 동작 없음 확인 |
| P1 | **1/200 검수(uncertain 1, negative 0), 199장 남음** | Vercel strict 완료 게이트 확인 |
| P2 | 검수 완료 전이라 실행 대기 | Mac→Windows 경로 분리 import와 전체 decode preflight 검증 |
| P3 | positive-only dev A/B·benchmark 완료, 최종 A/B는 정상 데이터 대기 | 동일 예산 A/B config·CUDA 4GB 확인·예약 실행·상태창 검증 |
| P4 | A/B 전이라 최종 평가는 대기, v0/v0.1 개발 validation 재평가 완료 | 19개 threshold, audit/boundary 검증, test 1회·완료 파일 복구 검증 |
| P5 | 최종 모델 전이라 대기, 기존 모델의 대형 BMP smoke benchmark 완료 | BMP runner·결과 집계·사람 검토표 생성 검증 |
| P6 | 파일럿 오류 전이라 실행 대기 | crop→재검수→누적 manifest→재학습→동일 A/B 품질 게이트·재파일럿 구현 |
| P7 | 데이터 기반 한계 미확인 | 지금은 실행하지 않음 |

## 우선순위 요약

| 우선순위 | 단계 | 예상 소요 | 다음 단계 진입 조건 |
|---|---|---:|---|
| P0 | 기준 산출물 고정과 원격 접속 마무리 | 20~30분 | 체크포인트 보존, Tailscale SSH 검증 |
| P1 | 정상/hard-negative 199장 검수 | 1~2시간 | 200/200 결정, 충돌 0 |
| P2 | 검수 결과 import와 데이터 감사 | 10~20분 | split별 정상 존재, 누수 0 |
| P3 | v0.2 A/B 학습 | 2~4시간 | 두 실험 정상 종료, best checkpoint 저장 |
| P4 | validation 보정과 held-out test | 30~60분 | recall 0.95 조건 충족, test 보고서 생성 |
| P5 | 대형 BMP 파일럿 | 1~2시간 | 현장 검토와 오류 목록 확보 |
| P6 | hard-negative 반복 학습 | 데이터에 따라 반복 | 주요 오탐 감소 |
| P7 | 모델 구조 변경 | 필요할 때만 | 기존 모델의 데이터 기반 한계 확인 |

예상 시간은 현재 RTX 3050 Laptop GPU와 기존 실행 시간을 기준으로 한 대략적인
작업 시간입니다. 사람 검수 속도와 대형 BMP 개수에 따라 달라집니다.

## P0. 기준 산출물 고정과 원격 접속 마무리

목적:

- 다음 실험이 기존 결과를 덮어쓰지 않게 함
- 노트북이 외부에 있어도 학습과 상태 확인이 가능하게 함

작업:

1. v0 best와 v0.1 best/last 체크포인트의 SHA-256 기록
2. 실행 config, source commit, manifest hash를 체크포인트와 함께 보존
3. Mac·Windows Tailscale을 같은 계정으로 로그인
4. `ourbrain-gpu-remote` SSH 별칭 생성
5. LAN이 아닌 Tailscale IP로 실제 SSH 재접속

완료 조건:

- 기존 v0/v0.1 결과가 별도 경로에 보존됨
- Windows 재부팅 후에도 Tailscale과 SSH가 자동 실행됨
- HDMI 없이 덮개를 닫은 상태에서 외부 SSH 성공

## P1. 정상/hard-negative 199장 검수

**전체 계획의 임계 경로이며 가장 높은 우선순위입니다.**

2026-07-31 원격 API 확인값은 revision 1, reviewed 1/200, uncertain 1,
negative 0, conflict 0입니다. 아직 어느 split에도 검수 정상 표본이 없습니다.

검수 URL:

```text
https://ourbrain-tunnel-review.vercel.app
```

판정:

- `N`: 균열 없음
- `C`: 균열 또는 균열 의심
- `U`: 판단 곤란

검수 기준:

- 작은 선이라도 균열 가능성이 있으면 `C`
- 구조물과 균열을 구분할 수 없으면 `U`
- 확실히 균열이 없을 때만 `N`
- 이음부, 케이블, 볼트, 얼룩, 누수와 반사를 정상 hard-negative로 확보

완료 조건:

- reviewed 200/200
- unreviewed 0
- conflict 0
- train/validation/test 각각에 `negative` 1장 이상

중단 조건:

- split 중 하나라도 정상 표본이 0이면 최종 학습으로 진행하지 않음

## P2. 검수 결과 import와 데이터 감사

작업:

```bash
uv run ourbrain-cv remote-review-download \
  --url https://ourbrain-tunnel-review.vercel.app \
  --output data/negative_review/negative_review_reviewed.csv

uv run ourbrain-cv import-negatives \
  --review data/negative_review/negative_review_reviewed.csv \
  --manifest artifacts/manifest.csv \
  --output artifacts/manifest_with_negatives.csv
```

검증:

1. review audit 완료 상태와 SHA-256
2. 전체 행 수와 split별 양성/정상 개수
3. group leakage 0
4. `crack`과 `uncertain`이 정상으로 들어가지 않았는지 확인
5. 실제 파일 디코딩 가능 여부

산출물:

- `artifacts/manifest_with_negatives.csv`
- `artifacts/manifest_with_negatives.review.json`
- 갱신된 데이터 통계 문서

Mac과 Windows의 데이터 루트가 다르므로 Mac manifest를 그대로 복사하지 않습니다.
`scripts/mac/finish_review_and_launch_v0_2.sh`가 검수 CSV와 후보 이미지를 전송한
뒤 `scripts/windows/import_v0_2_negatives.ps1`로 Windows base manifest에
strict import하고, 두 v0.2 config의 preflight를 모두 통과시킵니다.

## P3. v0.2 A/B 학습

두 실험 외 조건을 동일하게 유지합니다.

### 실험 A: 안정 기준선

- v0 objective
- 검수 정상 데이터 포함
- focal + Dice + boundary
- 최대 15 epoch, patience 6
- 설정: `configs/v0_2_a_baseline_with_negatives.yaml`

### 실험 B: recall 강화

- v0.1 sampling과 objective
- 검수 정상 데이터 포함
- crack-centered/background sampling
- focal + Tversky + clDice + boundary
- warmup/cosine과 backbone freeze
- 최대 15 epoch, patience 6
- 설정: `configs/v0_2_b_recall_with_negatives.yaml`

공통 통제 조건:

- 동일 manifest
- 동일 group split
- 동일 seed
- 동일 입력 크기와 학습 예산
- 동일 controlled augmentation
  - 좌우 반전 50%, 상하 반전 25%
  - 밝기·대비 ±20%, gamma ±15%
  - 작은 회전 ±8°
  - 약한 Gaussian blur 15%, Gaussian noise 20%
- 별도 output directory
- best와 last checkpoint 모두 저장

augmentation은 이미지와 mask의 기하 변환을 함께 적용하며, 색·명암·blur·noise는
RGB 이미지에만 적용합니다. 임의 90° 회전, 강한 perspective와 elastic 변형은
얇은 균열 정답을 훼손할 위험이 있어 v0.2에서는 제외합니다. A/B에 동일한
augmentation을 적용해 loss와 sampling 차이만 비교할 수 있게 유지합니다.

비교 지표:

- validation image recall/specificity
- crack Dice/IoU
- pixel precision/recall
- boundary F1
- 구조물 유형별 false positive

선택 규칙:

1. image recall 0.95 미만인 모델은 탈락
2. 통과 모델 중 image specificity 우선
3. 동률이면 crack Dice, boundary F1, 높은 threshold 순으로 비교
4. 최종 결정 전 overlay 오류 사례를 사람이 확인

## P4. validation 보정과 held-out test

선택된 각 후보 checkpoint에 대해 validation threshold를 보정합니다. 실제 runner는
0.05부터 0.95까지 0.05 간격을 비교합니다.

```bash
uv run ourbrain-cv calibrate \
  --config <config> \
  --checkpoint <checkpoint> \
  --minimum-image-recall 0.95 \
  --output <calibration.json>
```

보정 성공 후 threshold와 후처리를 변경하지 않고 test를 한 번 평가합니다.

```bash
uv run ourbrain-cv evaluate \
  --config <config> \
  --checkpoint <checkpoint> \
  --calibration <calibration.json> \
  --split test \
  --output <test_metrics.json>
```

완료 조건:

- calibration provenance 검증 성공
- 후보별 review audit SHA-256과 boundary tolerance 일치
- recall 제약 충족
- test metrics와 이미지 단위 false-positive/false-negative 파일 경로 저장
- test 결과를 보고 threshold를 다시 조정하지 않음
- test JSON에 config, checkpoint, manifest, review audit, calibration의 경로와
  SHA-256이 모두 기록됨
- `evaluation_complete.json`의 selection/test/config/checkpoint SHA-256 검증 성공

## P5. 대형 BMP 파일럿

대표 대형 BMP를 다음 범주에서 선정합니다.

- 명확한 균열
- 매우 얇거나 흐린 균열
- 정상 터널 벽
- 이음부와 케이블이 많은 이미지
- 오염, 누수, 조명 반사가 강한 이미지

확인 항목:

- 추론 시간과 메모리
- 타일 경계 artifact
- `maximum_positive_ratio` 게이트
- false positive와 false negative 위치
- overlay의 현장 해석 가능성

결과는 다음 hard-negative 수집 목록으로 연결합니다.

실행 파일:

- 입력 목록 예시: `configs/pilot_inputs.example.txt`
- 검증/실행: `scripts/windows/run_v0_2_pilot.ps1`
- headless 실행: `scripts/windows/launch_v0_2_pilot.ps1`

파일럿 runner는 P4의 모델 선택과 held-out test 결과가 모두 존재해야만 실행되며,
validation에서 고정한 threshold를 변경하지 않습니다. 각 overlay의 사람 판정은
`pilot_review.csv`에 남기고, 위치를 특정할 수 있는 오류는 원본 좌표까지 기록합니다.

## P6. Hard-negative 반복 학습

파일럿에서 반복적으로 틀리는 정상 영역을 추가 수집합니다.

반복:

1. 오탐 patch 수집
2. 사람 재검수
3. train/validation/test 역할에 맞게 group 단위 배치
4. 재학습
5. 같은 품질 게이트로 재평가

한 번에 많은 모델을 바꾸기보다 데이터 오류 유형 하나씩 줄입니다.
`ourbrain-cv pilot-hard-negatives`는 완료된 `pilot_review.csv`에서 좌표가 있는
false-positive만 crop하고 `review_label`을 비워 두어 두 번째 사람 검수를
강제합니다.

각 라운드는 안전한 `RoundId`로 분리합니다. 다음 스크립트가 이전 manifest의
review audit를 연결한 누적 manifest를 만들고, P4 선택 모델을 초기 checkpoint로
사용해 별도 디렉터리에 재학습합니다.

- `scripts/windows/import_hard_negative_round.ps1`
- `scripts/windows/run_hard_negative_round.ps1`
- `scripts/windows/launch_hard_negative_round.ps1`
- `scripts/windows/run_hard_negative_evaluation.ps1`
- `scripts/windows/launch_hard_negative_evaluation.ps1`

재평가는 새 manifest에서 기존 선택 모델과 재학습 모델을 모두 validation
calibration한 뒤 같은 recall 0.95 → specificity → Dice → boundary F1 정책으로
선택합니다. 선택된 하나만 새 held-out test에 한 번 적용하며, 이후
`run_v0_2_pilot.ps1 -RoundId <id>`로 같은 BMP 목록을 다시 실행합니다.

## P7. 모델 구조 변경

P1~P6을 수행한 뒤에도 목표를 충족하지 못할 때만 진행합니다.

RTX 3050 4GB 후보:

- SegFormer-B2
- DeepLabV3+
- UPerNet 입력 crop/해상도 조정

모델 변경 전 확인할 사항:

- 성능 한계가 정상 데이터 부족 때문이 아닌지
- 라벨 중심선 품질과 이미지 정합 문제가 없는지
- threshold와 후처리만으로 해결 가능한지
- VRAM과 대형 이미지 처리 시간이 허용되는지

## 즉시 실행 순서

1. Tailscale 사용자 승인과 외부 SSH 검증
2. 검수 링크에서 남은 199장 판정
3. 검수 완료 즉시 strict import
4. v0.2 A/B 학습
5. validation calibration
6. held-out test
7. 대형 BMP 파일럿

현재 당장 성능을 올리는 가장 중요한 작업은 **새 모델을 더 돌리는 것이 아니라
정상 199장을 정확히 검수하는 것**입니다.
