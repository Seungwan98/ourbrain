# 학습 및 실험 결과

최종 갱신: 2026-07-31

## 공통 환경

| 항목 | 값 |
|---|---|
| 모델 | `openmmlab/upernet-swin-tiny` |
| 작업 | 2-class semantic segmentation |
| 입력 | 512×512 |
| 클래스 | background / crack |
| CUDA 장비 | Lenovo, NVIDIA GeForce RTX 3050 Laptop GPU 4GB |
| batch size | 1 |
| gradient accumulation | 8 |
| mixed precision | 사용 |
| BatchNorm 통계 | 고정 |
| split | 원본 group 단위 train/val/test |
| 학습 데이터 | 검수 정상 데이터가 없는 양성 위주 1,223장 |

두 실험은 모두 exit code 0으로 정상 종료됐습니다. 다만 정상 데이터가 없으므로
모두 **개발용 체크포인트**이며 운영 배포 대상이 아닙니다.

## v0: 양성 기준 모델

목적은 Hugging Face UPerNet-Swin-Tiny의 전체 CUDA 파인튜닝 기준선을 만드는
것입니다.

| 설정 | 값 |
|---|---|
| 최대 epoch | 30 |
| 실제 완료 epoch | 22 |
| early stopping patience | 6 |
| learning rate | 6e-5 |
| loss | `1.0 focal + 1.0 Dice + 0.25 boundary` |
| 실행 시간 | 약 1시간 21분 |
| 최고 checkpoint | epoch 16 |

최고 validation crack Dice:

| 지표 | 값 |
|---|---:|
| crack Dice | **0.256632** |
| crack IoU | 0.149601 |
| precision | 0.168695 |
| recall | 0.560757 |
| pixel specificity | 0.995441 |
| boundary F1 | 0.834436 |

Windows 체크포인트:

```text
D:\ourbrain\runs\v0-positive-only\checkpoint
```

## v0.1: 중심선과 얇은 균열 강화 실험

v0.1의 가설은 다음과 같습니다.

1. 전체 이미지만 축소하지 않고 균열 중심 crop을 더 자주 보여주면 얇은 crack
   표현이 좋아질 수 있다.
2. Tversky로 false negative의 비용을 높이면 recall을 올릴 수 있다.
3. clDice로 중심선 연결성을 학습하면 끊긴 균열을 줄일 수 있다.
4. 이미 학습된 v0 backbone을 잠시 고정한 뒤 작은 learning rate로 안정적으로
   미세조정한다.

샘플링:

| 샘플 유형 | 비율 |
|---|---:|
| 전체 이미지 | 25% |
| 균열 중심 crop | 50% |
| 안전한 background crop | 25% |

학습 설정:

| 설정 | 값 |
|---|---|
| 시작 checkpoint | v0 best, epoch 16 |
| 최대 epoch | 15 |
| 실제 완료 epoch | **7** |
| 종료 방식 | patience 6에 의한 정상 early stopping |
| backbone freeze | epoch 1~2 |
| learning rate | 2e-5 |
| scheduler | 10% warmup + cosine decay |
| loss | `0.5 focal + 1.0 Tversky(α=0.7, β=0.3) + 0.5 clDice + 0.25 boundary` |
| 실행 시간 | 약 27분 |
| 최고 checkpoint | epoch 1 |

최고 validation crack Dice:

| 지표 | 값 |
|---|---:|
| crack Dice | **0.254753** |
| crack IoU | 0.148374 |
| precision | 0.164878 |
| recall | **0.584470** |
| pixel specificity | 0.995062 |
| boundary F1 | **0.834777** |

Windows 체크포인트:

```text
D:\ourbrain\runs\v0.1-sampling-tversky-cldice\checkpoint
D:\ourbrain\runs\v0.1-sampling-tversky-cldice\checkpoint\last
```

## v0와 v0.1 비교

각 실험의 validation crack Dice가 가장 높은 epoch를 비교합니다.

| 지표 | v0 epoch 16 | v0.1 epoch 1 | 변화 |
|---|---:|---:|---:|
| crack Dice | **0.256632** | 0.254753 | -0.001879 (-0.73%) |
| crack IoU | **0.149601** | 0.148374 | -0.001228 |
| precision | **0.168695** | 0.164878 | -0.003818 |
| recall | 0.560757 | **0.584470** | +0.023713 |
| pixel specificity | **0.995441** | 0.995062 | -0.000379 |
| boundary F1 | 0.834436 | **0.834777** | +0.000340 |

## 표준화된 개발 성능 재평가

2026-07-31에 두 best checkpoint를 동일한 추론 설정으로 validation 221장 전체에
다시 적용했습니다. 이 값은 학습 history의 최고 epoch 기록이 아니라,
`threshold=0.5`, `minimum_component_pixels=8`, `boundary_tolerance=2`를 동일하게
적용한 별도 GPU 재평가 결과입니다. held-out test split은 열지 않았습니다.

| 지표 | v0 | v0.1 | 변화(v0.1-v0) |
|---|---:|---:|---:|
| crack Dice | **0.257609** | 0.253832 | -0.003776 |
| crack IoU | **0.147848** | 0.145365 | -0.002482 |
| precision | **0.167514** | 0.162345 | -0.005170 |
| recall | 0.557392 | **0.581572** | +0.024180 |
| pixel specificity | **0.995452** | 0.995074 | -0.000379 |
| boundary F1 | 0.832424 | **0.834045** | +0.001621 |
| 처리 시간, 221장 | 38.08초 | **37.47초** | -0.61초 |
| 처리량 | 5.80장/초 | **5.90장/초** | +0.09장/초 |

이미지별 Dice의 paired 비교에서는 v0가 128장, v0.1이 92장에서 높았고 1장은
동률이었습니다. 이미지별 Dice 차이 `v0-v0.1`의 평균은 `+0.001829`이며,
고정 seed 10,000회 paired bootstrap 95% 구간은
`[+0.000579, +0.003146]`입니다. 따라서 현재 양성 validation에서는 v0의
작은 Dice 우위가 일관되지만, 정상 이미지 오탐 성능을 뜻하지는 않습니다.

원본 대형 BMP `Tube_009_1.bmp`(1798×10000, 17.98MP)도 Windows CUDA에서
타일 추론했습니다.

| 항목 | v0 | v0.1 |
|---|---:|---:|
| 전체 실행 시간 | 21.147초 | 21.161초 |
| 처리량 | 0.85MP/초 | 0.85MP/초 |
| 예측 균열 비율 | 0.20498% | 0.24658% |
| 연결 성분 수 | 168 | 185 |
| quality gate | 통과 | 통과 |

두 모델 모두 균열 존재로 판정했습니다. 다만 이 BMP에는 신뢰할 수 있는 정답
마스크가 없으므로 정확도 판정이 아니라 실행 속도와 출력 형태 확인 결과입니다.
overlay 육안 확인상 v0.1이 수평 이음부와 표면 선을 더 많이 표시했으며, 이는
recall 증가와 동시에 false-positive 위험이 커졌을 가능성을 보여줍니다.

Windows 원본 결과:

```text
D:\ourbrain\runs\development-benchmark\benchmark_complete.json
D:\ourbrain\runs\development-benchmark\large-bmp\large_bmp_complete.json
```

## 해석과 결정

- v0.1은 학습 history에서 recall을 약 2.37%p, 표준 재평가에서 약 2.42%p
  높였습니다.
- boundary F1은 사실상 동일하며 소폭 상승했습니다.
- precision이 낮아졌고 주 선택 지표인 crack Dice는 0.73% 하락했습니다.
- v0.1의 최고 점수가 첫 epoch에서 나온 뒤 개선되지 않아 epoch 7에서 정상
  조기 종료됐습니다.

따라서 현재 결정은 다음과 같습니다.

1. **v0 epoch 16을 개발 기준 체크포인트로 유지**합니다.
2. v0.1은 최종 승격하지 않고 recall 중심 학습법의 실험 결과로 보존합니다.
3. v0.1의 `46.7%` 표시는 15 epoch 중 7 epoch에서 early stopping된 비율이며,
   멈춤이나 오류가 아닙니다.
4. 검수된 정상 데이터가 추가되기 전에는 두 모델의 이미지 단위 오탐 성능을
   비교할 수 없습니다.

loss 정의가 서로 다르므로 v0와 v0.1의 `val_loss` 숫자는 직접 비교하지 않습니다.
또한 양성 데이터만 있는 validation에서 이미지 단위 specificity가 `1.0`으로
표시되는 것은 정상 표본이 없기 때문에 의미가 없습니다.

## v0.2 augmentation 준비

v0.2 A/B에는 동일한 controlled augmentation을 설정했습니다.

- 좌우 반전 50%, 상하 반전 25%
- 밝기·대비 ±20%, gamma ±15%
- 작은 회전 ±8°
- Gaussian blur 15%(최대 radius 1.0)
- Gaussian noise 20%(표준편차 0.015)

회전과 반전은 이미지와 mask에 동일하게 적용하며 mask에는 nearest-neighbor
보간을 사용합니다. 실제 train 이미지와 mask로 7개 고정 seed 결과를 생성해
마스크 정렬과 변형 강도를 육안 확인했습니다. validation/test에는 이 변형을
적용하지 않습니다.

A/B 모두 같은 augmentation을 사용하므로 두 실험의 비교 차이는 기존 계획대로
sampling과 loss 구성에서 발생합니다. 임의 90° 회전, 강한 perspective와 elastic
변형은 얇은 균열을 인위적으로 훼손할 가능성이 있어 제외했습니다.

## v0.2-dev positive-only 사전 실험

정상 200장 검수 완료 전에도 augmentation과 학습 recipe를 검증하기 위해 최종
v0.2와 완전히 분리된 개발 A/B를 2026-07-31에 실행했습니다.

| 실험 | 확인 대상 | Windows output |
|---|---|---|
| dev A | v0 loss + controlled augmentation | `runs\v0.2-dev-a-augmentation-positive-only` |
| dev B | recall loss/sampling + controlled augmentation | `runs\v0.2-dev-b-augmentation-recall-positive-only` |

두 실험은 기존 v0 best에서 시작하며 동일한 positive-only manifest, seed, 입력
크기, augmentation과 15 epoch 예산을 사용합니다. 모든 결과에는
`development_only=true`, `production_eligible=false`,
`held_out_test_opened=false`를 기록합니다.

학습 결과:

| 실험 | 완료 epoch | best epoch | history Dice | history recall | history boundary F1 |
|---|---:|---:|---:|---:|---:|
| dev A | 12/15, 조기 종료 | 6 | 0.256057 | 0.540109 | **0.838975** |
| dev B | 7/15, 조기 종료 | 1 | 0.255152 | **0.561601** | 0.834499 |

post-training task가 두 best checkpoint를 기존 모델과 동일한 조건으로 다시
평가했습니다.

| 모델 | crack Dice | precision | recall | boundary F1 | 처리량 |
|---|---:|---:|---:|---:|---:|
| v0 | 0.257609 | 0.167514 | 0.557392 | 0.832424 | 5.80장/초 |
| v0.1 | 0.253832 | 0.162345 | **0.581572** | 0.834045 | **5.90장/초** |
| dev A | **0.258174** | **0.169812** | 0.538256 | **0.837484** | 5.80장/초 |
| dev B | 0.254591 | 0.164825 | 0.559066 | 0.833345 | 5.87장/초 |

dev A는 aggregate Dice에서 v0보다 `+0.000565`, boundary F1은 `+0.005061`
높았지만 recall은 `-0.019136` 낮았습니다. 이미지별 Dice 차이 `dev A-v0`의
평균은 `-0.000400`이고 10,000회 paired bootstrap 95% 구간은
`[-0.002959, +0.002134]`입니다. 13개 validation group 단위 bootstrap 구간도
`[-0.001825, +0.002827]`로 0을 포함합니다. 따라서 augmentation이 boundary와
precision에 유리한 신호는 있으나 v0보다 확실히 우수하다고 판정할 근거는 없습니다.

dev B는 v0.1보다 aggregate Dice가 `+0.000758` 높지만 recall이 `-0.022506`
낮아 recall 강화의 추가 이점이 확인되지 않았습니다. v0 대비 group bootstrap
Dice 차이는 `[-0.002338, -0.000524]`로 음수였습니다.

17.98MP 대형 BMP 결과:

| 모델 | 실행 시간 | 균열 예측 비율 | 연결 성분 |
|---|---:|---:|---:|
| v0 | 21.147초 | 0.20498% | 168 |
| v0.1 | 21.161초 | 0.24658% | 185 |
| dev A | 21.136초 | 0.22149% | 224 |
| dev B | **21.102초** | 0.22116% | 210 |

모든 모델이 균열 존재로 판정했고 처리량은 약 0.85MP/초로 동일했습니다. 이 BMP는
정답 mask가 없으므로 연결 성분 증가를 개선으로 해석할 수 없습니다. overlay에는
수평 이음부와 표면 선으로 보이는 검출도 있어 정상 데이터 기반 검증이 필요합니다.

현재 사전 결정:

1. dev A augmentation은 최종 v0.2 A/B 공통 설정으로 유지
2. positive-only 기준 체크포인트는 통계적으로 확실한 우위가 없는 v0 유지
3. dev B의 recall recipe는 정상 데이터가 들어온 최종 B에서 다시 검증
4. 검수 정상 데이터가 도착하면 final v0.2 A/B를 별도 경로에서 재학습

Windows 완료 결과:

```text
D:\ourbrain\runs\v0.2-dev-positive-only-ab\training_complete.json
D:\ourbrain\runs\v0.2-dev-benchmark\benchmark_complete.json
```

## v0.3: 모델 아키텍처 비교

정상 200장이 도착하기 전 positive-only 데이터에서 모델 자체를 바꿨을 때의
효과를 확인하기 위해 2026-08-01~02에 세 후보를 같은 조건으로 학습했습니다.
이 실험은 최종 모델 선정이 아니라 개발 후보 제거를 위한 비교입니다.

| ID | 아키텍처 | Hugging Face 시작 체크포인트 |
|---|---|---|
| A | UPerNet + Swin-Tiny | `openmmlab/upernet-swin-tiny` |
| B | SegFormer-B1 | `nvidia/segformer-b1-finetuned-ade-512-512` |
| C | SegFormer-B2 | `nvidia/segformer-b2-finetuned-ade-512-512` |

세 후보는 모델 이외의 설정을 고정했습니다. 입력 512, batch 1, gradient
accumulation 8, 최대 30 epoch, backbone freeze 2 epoch, AdamW
`lr=6e-5`, `weight_decay=0.01`, cosine warmup 10%, FP16을 사용했습니다.
augmentation은 flip, 밝기·대비·gamma, ±8° 회전, blur와 Gaussian noise를
포함합니다. loss는 Focal 1.0 + Dice 1.0 + Boundary 0.25 + Tversky 0.25
(`alpha=0.3`, `beta=0.7`) + clDice 0.15입니다.

GPU 스모크 테스트는 각 모델을 backbone unfrozen 상태로 10 step 실행했습니다.

| 모델 | 스모크 peak CUDA | 전체 학습 | best epoch | 전체 학습 peak CUDA |
|---|---:|---:|---:|---:|
| A UPerNet | 1.891GiB | 30/30 | 27 | 1.890GiB |
| B SegFormer-B1 | 0.628GiB | 26/30, 조기 종료 | 20 | 0.630GiB |
| C SegFormer-B2 | 1.337GiB | 30/30 | 26 | 1.338GiB |

세 checkpoint를 기존 v0와 같은 validation 221장, threshold 0.5,
boundary tolerance 2 조건으로 다시 평가했습니다.

| 모델 | crack Dice | precision | recall | boundary F1 | 처리량 |
|---|---:|---:|---:|---:|---:|
| v0 기준 | **0.257609** | **0.167514** | 0.557392 | **0.832424** | 5.80장/초 |
| A UPerNet | 0.250571 | 0.161583 | **0.557718** | 0.809488 | 5.84장/초 |
| B SegFormer-B1 | 0.237123 | 0.155270 | 0.501490 | 0.789967 | **11.27장/초** |
| C SegFormer-B2 | 0.244192 | 0.157631 | 0.541615 | 0.815838 | 8.37장/초 |

validation의 13개 group을 단위로 seed 42, 10,000회 paired bootstrap을
수행했습니다.

| 후보 | Dice 차이(후보-v0) | group bootstrap 95% CI | v0 대비 recall 감소 | gate |
|---|---:|---:|---:|---|
| A | -0.007038 | [-0.014005, -0.006208] | -0.000326 | 실패 |
| B | -0.020486 | [-0.026423, -0.017359] | 0.055901 | 실패 |
| C | -0.013417 | [-0.018169, -0.008394] | 0.015776 | 실패 |

세 신뢰구간이 모두 0보다 작아 새 후보의 Dice가 v0보다 낮다는 방향이
일관됐습니다. B는 속도와 메모리 효율은 가장 좋았지만 Dice·recall·boundary
F1이 모두 가장 낮았고 recall 감소 허용치 0.02도 넘었습니다. 따라서
positive-only 기준 모델은 `v0-positive-only`를 유지합니다.

17.98MP `Tube_009_1.bmp` 스모크 결과:

| 모델 | 실행 시간 | 처리량 | 균열 예측 비율 | 연결 성분 |
|---|---:|---:|---:|---:|
| A UPerNet | 21.151초 | 0.850MP/초 | 0.21626% | 308 |
| B SegFormer-B1 | **10.077초** | **1.784MP/초** | 0.27936% | 515 |
| C SegFormer-B2 | 18.116초 | 0.993MP/초 | 0.21175% | 494 |

세 결과 모두 파일·품질 gate를 통과했습니다. overlay 육안 검토에서 B는
수평 표면 무늬와 이음부 검출이 A/C보다 많았습니다. 이 BMP에는 정답 mask가
없으므로 속도와 출력 형상만 확인할 수 있고 정확도 비교 근거로는 사용하지
않습니다.

완료 결과:

```text
D:\ourbrain\runs\v0.3-model-sweep\training_complete.json
D:\ourbrain\runs\v0.3-model-sweep\benchmark\benchmark_complete.json
D:\ourbrain\runs\v0.3-model-sweep\benchmark\development_selection.json
```

검증 계약은 `development_only=true`, `production_eligible=false`,
`positive_only=true`, `held_out_test_opened=false`입니다. 사람 검수 정상 데이터가
없어 운영 false-positive specificity는 아직 측정할 수 없습니다.

## 재현 설정

저장소 설정:

```text
configs/upernet_swin_tiny.yaml
configs/v0_1_sampling_tversky_cldice.yaml
configs/v0_3_a_upernet_swin_tiny_positive_only.yaml
configs/v0_3_b_segformer_b1_positive_only.yaml
configs/v0_3_c_segformer_b2_positive_only.yaml
```

v0.1 실행 예:

```bash
uv run ourbrain-cv train \
  --config configs/v0_1_sampling_tversky_cldice.yaml \
  --manifest artifacts/manifest.csv \
  --allow-positive-only \
  --device cuda
```

`--allow-positive-only`는 개발 실험용 우회 옵션입니다. 이 옵션으로 생성한
체크포인트를 최종 모델로 간주하면 안 됩니다.

## 다음 학습 실험

정상 200장 검수가 끝난 뒤 동일 데이터와 동일 split으로 다음 A/B를 수행합니다.

- A: v0 loss + 검수 정상 데이터
- B: v0.1 sampling/loss + 검수 정상 데이터

두 실험은 각각
`configs/v0_2_a_baseline_with_negatives.yaml`과
`configs/v0_2_b_recall_with_negatives.yaml`에 고정돼 있습니다. 모델 시작점,
최대 epoch, learning rate, freeze 기간, scheduler와 추론 설정은 동일하고
sampling과 loss 구성만 비교합니다.

선택 기준은 validation의 pixel Dice 하나가 아니라 다음을 함께 봅니다.

1. 이미지 단위 recall 하한 0.95 충족
2. 정상 validation에서의 이미지 단위 specificity
3. crack Dice/IoU
4. boundary F1
5. 터널 구조물별 false positive 사례

threshold를 validation에서 고정한 뒤 held-out test를 한 번 평가합니다.
