# 개발·학습·원격 운영 절차

## 로컬 개발 환경

요구사항:

- Python 3.11 또는 3.12
- `uv`
- macOS MPS 또는 Windows CUDA

설치와 검증:

```bash
uv sync --extra dev
uv run ourbrain-cv --help
uv run pytest
uv run ruff check .
```

## 저장소와 장비 역할

Windows GPU 노트북을 프로젝트의 기준 실행 장비로 사용합니다.

```text
D:\ourbrain                  코드, 설정, 문서, 검토 도구
D:\ourbrain\runs             실험별 로그와 중간 산출물
D:\ourbrain\checkpoints      보존할 best/last 모델과 메타데이터
D:\ourbrain-data             실행용 학습·검증 데이터 복사본
E:\train, E:\bmp             외장하드 원본 데이터(읽기 위주 보관)
E:\ourbrain-backup           중요 모델·결과의 장비 외 백업
```

원본 외장하드에서 직접 학습하거나 파일명을 변경하지 않습니다. 유효한 쌍만
`D:\ourbrain-data`로 복사해 실행하고, 모델과 실험 결과의 기준본은 Windows에
둡니다. Git은 소스·설정·테스트·문서·웹 코드 공유에 사용하고 데이터, 가상환경,
실험 결과와 대용량 모델은 Git에 넣지 않습니다.

Mac과 다른 협업 장비는 Git 복제본과 SSH/Tailscale 접속 도구만 있으면 됩니다.
대용량 모델 전달은 Windows의 SSH/SFTP 경로를 사용합니다. 다만 장비 고장에
대비해 최종 모델과 선택 근거 JSON은 외장하드에도 한 벌 보존합니다.

## Mac 역할

Mac은 다음 작업을 보조할 수 있지만 기준 저장소는 Windows입니다.

- 외장 디스크 데이터 감사
- manifest와 검수 후보 생성
- 로컬/Vercel 검수 관리
- MPS smoke test
- 결과 분석과 문서화

MPS smoke test는 실행 경로 확인용이며 실제 성능 학습이 아닙니다.

```bash
uv run ourbrain-cv train \
  --config configs/smoke_mps.yaml \
  --device mps \
  --allow-positive-only
```

## Windows GPU 노트북 역할

경로:

```text
프로젝트: D:\ourbrain
실험 결과: D:\ourbrain\runs
실행/모니터 스크립트: D:\ourbrain-bootstrap
GPU: NVIDIA GeForce RTX 3050 Laptop GPU 4GB
```

같은 LAN에서 Mac의 SSH 별칭:

```bash
ssh ourbrain-gpu
```

v0.1 상태창:

```powershell
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain-bootstrap\watch_v01_improved.ps1
```

완료 상태는 다음처럼 표시됩니다.

```text
STATUS : COMPLETED - EARLY STOPPING AT EPOCH 7/15
Result : NORMAL COMPLETION by early stopping
ETA    : finished
```

`46.7%`는 최대 epoch 대비 사용한 비율이며, 완료 파일의 exit code가 0이면 오류가
아닙니다.

현재 v0와 v0.1을 동일한 validation 조건으로 비교하는 개발 benchmark:

```powershell
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\run_development_benchmark.ps1
```

이 benchmark는 validation만 사용하며 held-out test를 열지 않습니다. 결과는
`D:\ourbrain\runs\development-benchmark\benchmark_complete.json`에 기록됩니다.
완료 결과가 있으면 덮어쓰지 않습니다.

정답 마스크가 없는 대표 대형 BMP의 속도와 출력 형태를 확인하는 smoke benchmark:

```powershell
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\run_development_large_bmp_benchmark.ps1 `
  -InputImage D:\ourbrain-data\performance-benchmark\Tube_009_1.bmp
```

이 결과의 균열 비율은 정확도 근거가 아닙니다. 정답 마스크가 없으므로 overlay를
사람이 확인해야 하며, 정상 데이터 오탐률이나 운영 적합성을 주장할 수 없습니다.

v0.2 A/B는 검수된 정상 manifest가 준비된 뒤 다음 스크립트로 실행합니다.

```powershell
# 사람 검수와 strict import가 끝났는지만 확인
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\run_v0_2_ab.ps1 `
  -PreflightOnly

# headless 예약 작업으로 A → B 순차 실행
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\launch_v0_2_ab.ps1

# 진행 상황 확인
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\watch_v0_2_ab.ps1
```

runner는 `--allow-positive-only`를 사용하지 않습니다. 검수 manifest나 audit가
없거나 split별 정상 샘플이 부족하면 GPU 학습을 시작하기 전에 중단합니다.

정상 200장 도착 전에 augmentation과 recipe만 검증하는 개발 A/B는 별도 경로를
사용합니다.

```powershell
# positive-only 전용 gate와 전체 파일 decode만 확인
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\run_v0_2_dev_positive_ab.ps1 `
  -PreflightOnly

# headless 개발 A/B 시작
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\launch_v0_2_dev_positive_ab.ps1

# 실시간 상태
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\watch_v0_2_dev_positive_ab.ps1

# 학습 완료를 기다렸다가 validation + 대형 BMP 비교 자동 실행
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\launch_v0_2_dev_posttrain.ps1
```

개발 runner는 `--allow-positive-only`를 명시적으로 사용하지만 final v0.2 runner와
config/output/task 이름이 모두 다릅니다. 검수 정상이 이미 manifest에 존재하면
오히려 개발 runner가 중단되며, held-out test는 열지 않습니다.
post-training task는 `training_complete.json`을 기다린 뒤 v0/v0.1/dev A/dev B를
동일한 positive validation 조건으로 비교하고, 같은 17.98MP BMP에서 dev A/B
추론을 실행합니다. 결과는
`D:\ourbrain\runs\v0.2-dev-benchmark\benchmark_complete.json`에 기록됩니다.

v0.2 A/B config는 동일한 controlled augmentation을 사용합니다.

```yaml
data:
  augmentation:
    horizontal_flip_probability: 0.5
    vertical_flip_probability: 0.25
    brightness_jitter: 0.2
    contrast_jitter: 0.2
    rotation_degrees: 8.0
    gamma_jitter: 0.15
    gaussian_blur_probability: 0.15
    gaussian_blur_radius: 1.0
    gaussian_noise_probability: 0.2
    gaussian_noise_std: 0.015
```

seed가 같으면 동일한 변형 순서를 재현합니다. validation/test에는 augmentation을
적용하지 않고 resize와 normalization만 적용합니다.

A/B 학습이 모두 끝난 뒤 validation 보정과 단일 held-out test는 다음으로
실행합니다.

```powershell
# A/B 체크포인트 존재와 test 미실행 상태만 확인
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\run_v0_2_evaluation.ps1 `
  -PreflightOnly

# A/B 모두 보정 → validation 정책으로 1개 선택 → test 한 번 평가
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\launch_v0_2_evaluation.ps1
```

`held_out_test_metrics.json` 또는 `evaluation_complete.json`이 이미 존재하면
evaluation runner는 재실행을
거부합니다. test 결과를 본 뒤 threshold를 다시 조정하는 것을 막기 위한
의도적인 보호 장치입니다. 결과는 임시 파일에서 완성한 뒤 원자적으로 교체되며,
config·checkpoint·manifest·review audit·calibration 경로와 SHA-256을 함께
기록합니다. 전체 provenance 검증까지 끝나야 `evaluation_complete.json`이
생성되며, 파일럿은 이 완료 파일과 실제 결과의 SHA-256이 일치할 때만 시작합니다.
후보 선택 시 review audit가 바뀌었거나 boundary tolerance가 서로 다르면
validation 지표를 비교하지 않고 중단합니다.

test JSON 저장 직후 전원 중단 등으로 `evaluation_complete.json`만 빠진 경우에는
같은 evaluation runner를 다시 실행합니다. runner는 기존 test JSON과 selection,
calibration, manifest, review audit의 해시를 검증한 뒤 완료 파일만 복구하며 test
split을 다시 평가하지 않습니다. 두 파일이 모두 있으면 정상 완료 상태로 보고
재실행을 거부합니다.

held-out test가 끝난 뒤 대형 BMP 파일럿은 다음으로 준비하고 실행합니다.

```powershell
# 예시 파일을 복사한 뒤 실제 대표 BMP 절대 경로로 모두 교체
Copy-Item D:\ourbrain\configs\pilot_inputs.example.txt `
  D:\ourbrain\artifacts\pilot_inputs.txt
notepad D:\ourbrain\artifacts\pilot_inputs.txt

# 선택 모델·test 결과·BMP 목록을 읽기만 하고 검증
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\run_v0_2_pilot.ps1 `
  -PreflightOnly

# 덮개를 닫아도 유지되는 예약 작업으로 파일럿 실행
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\launch_v0_2_pilot.ps1
```

파일럿은 `model_selection.json`에 기록된 모델과 validation threshold를 그대로
사용합니다. 입력마다 probability, mask, overlay, summary를 저장하고 전체 결과를
`D:\ourbrain\runs\v0.2-pilot\pilot_summary.json`에 모읍니다. 중간 결과가 일부만
존재하면 덮어쓰지 않고 중단하며, 완료된 입력은 재실행 시 건너뜁니다. 파일럿 완료는
자동 운영 승인 뜻이 아니며 사람이 overlay에서 오탐·미탐을 기록해야 합니다.

사람 검토표는 `D:\ourbrain\runs\v0.2-pilot\pilot_review.csv`에 생성됩니다.
`review_label`은 `correct_crack`, `correct_normal`, `false_positive`,
`false_negative`, `uncertain` 중 하나로 채웁니다. 위치를 특정할 수 있는 오류는
원본 BMP 좌표 `left, top, right, bottom`도 기록해야 다음 hard-negative crop으로
연결할 수 있습니다. 기존 검토표는 재실행해도 덮어쓰지 않습니다.

검토표가 전부 채워진 뒤, 사람이 좌표까지 확인한 `false_positive`만 두 번째
검수용 512×512 crop으로 만듭니다.

```powershell
$round = 'round-001'

# 1. 좌표가 있는 false positive를 별도 crop으로 생성
D:\ourbrain\.venv\Scripts\python.exe -m ourbrain_cv.cli `
  pilot-hard-negatives `
  --review D:\ourbrain\runs\v0.2-pilot\pilot_review.csv `
  --output "D:\ourbrain\data\pilot_hard_negatives\$round"

# 2. hard_negative_review.csv의 모든 review_label을 사람이 다시 판정한 뒤 import
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\import_hard_negative_round.ps1 `
  -RoundId $round

# 3. 전체 decode와 CUDA를 읽기 전용으로 재검증
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\run_hard_negative_round.ps1 `
  -RoundId $round -PreflightOnly

# 4. 별도 예약 작업으로 재학습
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\launch_hard_negative_round.ps1 `
  -RoundId $round

# 5. 기존 모델과 재학습 모델을 같은 validation 정책으로 비교 후 test 1회
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\launch_hard_negative_evaluation.ps1 `
  -RoundId $round

# 6. 같은 대표 BMP 목록으로 재파일럿
powershell -ExecutionPolicy Bypass `
  -File D:\ourbrain\scripts\windows\launch_v0_2_pilot.ps1 `
  -RoundId $round
```

라운드 import는 crop SHA-256, 두 번째 검수 완료, 이전 manifest audit chain,
P4 선택 모델 해시를 확인합니다. 출력 manifest와 학습·평가·파일럿 경로에는
`RoundId`가 포함되며 기존 라운드를 덮어쓰지 않습니다. 새 test JSON 저장 직후
중단된 경우에도 evaluation runner는 해시 검증 후 완료 파일만 복구하고 test를
다시 실행하지 않습니다.

## 덮개를 닫은 상태의 운영

2026-07-31에 Windows의 **전원 연결(AC) 상태**를 다음과 같이 설정하고 실제 SSH
재접속으로 검증했습니다.

- 덮개 닫기: 아무 동작 안 함
- 자동 절전: 사용 안 함
- 자동 최대절전: 사용 안 함
- Wi-Fi 절전: 최대 성능
- 배터리 덮개 닫기: 절전(`LIDACTION DC=1`) 유지

따라서 HDMI 또는 외부 모니터 없이 덮개를 닫아도 SSH와 백그라운드 작업을 사용할
수 있습니다. 조건은 전원 어댑터와 네트워크가 연결돼 있어야 한다는 것입니다.
AC 덮개 동작은 `LIDACTION AC=0`으로 재검증했습니다. `sshd`와 `Tailscale`
Windows 서비스도 모두 `Running / Auto`입니다.

GPU 학습 중에는 화면이 필요하지 않지만 발열을 위해 통풍구를 막지 말고, 가능하면
거치대 위에 두거나 덮개를 열어두는 편이 안전합니다.

현재 설정 검증:

```powershell
powercfg /getactivescheme
powercfg /query SCHEME_CURRENT
```

## Tailscale 외부 접속

Tailscale 앱은 Mac과 Windows에 설치돼 있습니다. 2026-07-31 현재 Windows는
로그인 완료, `Running`, health 오류 0이며 Tailscale IP는 `100.92.39.77`입니다.
`ourbrain-gpu-remote` SSH 별칭도 이 IP로 구성했습니다. 남은 작업은 Mac의
Network Extension 사용자 보안 승인과 실제 외부 SSH 확인입니다.

- Mac: Tailscale Network Extension 승인
- Mac: Windows와 같은 tailnet에 로그인
- Mac: LAN route가 아닌 Tailscale IP로 SSH 접속 테스트

기존 LAN 별칭과 외부용 별칭은 다음처럼 분리돼 있습니다.

```sshconfig
Host ourbrain-gpu-remote
    HostName 100.92.39.77
    User LENOVO
    IdentityFile ~/.ssh/id_ed25519_ourbrain
```

다른 Mac에서도 Tailscale에 같은 계정으로 로그인하고, 해당 Mac의 SSH public key를
Windows `authorized_keys`에 추가하면 접속할 수 있습니다. private key를 메신저나
공유 드라이브로 복사하는 방식은 권장하지 않습니다.

## Vercel 정상 후보 검수

```text
https://ourbrain-tunnel-review.vercel.app
```

상태 확인:

```bash
set -a
source .env.remote-review.local
set +a

uv run ourbrain-cv remote-review-status \
  --url https://ourbrain-tunnel-review.vercel.app \
  --summary-only
```

완료 결과 다운로드:

```bash
uv run ourbrain-cv remote-review-download \
  --url https://ourbrain-tunnel-review.vercel.app \
  --output data/negative_review/negative_review_reviewed.csv
```

접근 토큰은 Git에 커밋하지 않습니다. Vercel 앱은 정상 후보 검수용이며 대용량 모델
학습이나 GPU 추론을 수행하지 않습니다.

검수가 200/200 완료된 뒤 Mac과 Windows의 경로 차이를 안전하게 처리하려면 다음
연결 스크립트를 사용합니다.

```bash
# 상태 확인 → 양쪽 strict import와 preflight까지만 수행
bash scripts/mac/finish_review_and_launch_v0_2.sh

# 같은 검증이 모두 통과했을 때 Windows A/B 예약 작업도 시작
bash scripts/mac/finish_review_and_launch_v0_2.sh --launch-training
```

Mac의 `/Volumes/...` manifest를 Windows로 그대로 복사하지 않습니다. 연결
스크립트는 검수 CSV와 200개 후보 PNG를 전송한 뒤, Windows의
`D:\ourbrain-data\...` base manifest를 기준으로 다시 import합니다. 검수가
미완료이거나 충돌이 있으면 CSV 다운로드·manifest 변경·GPU 학습을 전부 막습니다.

## 기본 파이프라인

```bash
# 1. 데이터 감사
uv run ourbrain-cv prepare \
  --data-root '/Volumes/새 볼륨/train' \
  --manifest artifacts/manifest.csv \
  --audit artifacts/data_audit.json

# 2. 정상 후보 생성
uv run ourbrain-cv negative-candidates \
  --raw-root '/Volumes/새 볼륨/bmp' \
  --manifest artifacts/manifest.csv \
  --output data/negative_review \
  --max-candidates 200

# 3. 검수 결과 반영
uv run ourbrain-cv import-negatives \
  --review data/negative_review/negative_review_reviewed.csv \
  --manifest artifacts/manifest.csv \
  --output artifacts/manifest_with_negatives.csv

# 4. 최종 학습
uv run ourbrain-cv training-preflight \
  --config configs/v0_2_a_baseline_with_negatives.yaml \
  --require-local-checkpoint \
  --verify-files \
  --device cuda

uv run ourbrain-cv train --config configs/upernet_swin_tiny.yaml

# 5. validation 임계값 선택
uv run ourbrain-cv calibrate \
  --config configs/upernet_swin_tiny.yaml \
  --checkpoint checkpoints/upernet-swin-tiny \
  --minimum-image-recall 0.95 \
  --output artifacts/threshold_calibration.json

# 6. held-out test 한 번 평가
uv run ourbrain-cv evaluate \
  --config configs/upernet_swin_tiny.yaml \
  --checkpoint checkpoints/upernet-swin-tiny \
  --calibration artifacts/threshold_calibration.json \
  --split test \
  --output artifacts/test_metrics.json
```

## 문제 진단

### 상태창이 특정 비율에서 멈춘 것처럼 보임

`exit_code.txt`가 0이고 `finished_at.txt`가 존재하면 정상 종료입니다. early stopping은
최대 epoch에 도달하지 않아도 성공입니다.

### SSH 접속 불가

1. 노트북 전원과 Wi-Fi 확인
2. 같은 LAN이면 `192.168.123.103` 도달 여부 확인
3. 외부라면 양쪽 Tailscale 로그인 여부 확인
4. Windows `sshd`와 Tailscale 서비스 확인

### CUDA out of memory

- batch size 1 유지
- gradient accumulation으로 effective batch 확보
- 다른 GPU 프로세스 종료
- 입력 크기나 모델을 바꾸기 전에 현재 checkpoint 보존

### 학습 명령이 정상 데이터 부족으로 중단

의도된 품질 게이트입니다. `--allow-positive-only`로 최종 학습을 우회하지 말고 200장
검수를 완료해야 합니다.
