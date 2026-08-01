# OurBrain Tunnel Crack CV 문서

최종 갱신: 2026-08-02

이 디렉터리는 터널 스캔 이미지의 균열을 검출하는 OurBrain CV 프로젝트의
데이터, 설계, 학습 결과, 운영 절차와 다음 단계를 설명합니다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| 데이터 감사 및 누수 방지 manifest | 완료 |
| 양성 마스크 기반 v0 기준 모델 | 완료 |
| 샘플링·Tversky·clDice를 적용한 v0.1 실험 | 완료 |
| v0/v0.1 동일 조건 validation 성능 재평가 | 완료 |
| 17.98MP BMP 타일 추론 smoke benchmark | 완료 |
| v0.2 controlled augmentation 구현·실이미지 확인 | 완료 |
| v0.2-dev positive-only augmentation A/B·benchmark | 완료 |
| v0.3 UPerNet/SegFormer-B1/B2 모델 비교·benchmark | 완료, 기존 v0 유지 |
| 정상/hard-negative 후보 200장 생성 | 완료 |
| 정상 후보 사람 검수 | **1/200, uncertain 1장, negative 0장, 199장 남음** |
| 검수된 정상 데이터 포함 최종 학습 | 대기 |
| val 임계값 보정 및 held-out test 평가 | 대기 |
| 운영 배포 가능 모델 | **아직 없음** |

동일 조건으로 validation 221장을 다시 평가한 결과 v0의 crack Dice는
`0.257609`, v0.1은 `0.253832`, augmentation dev A는 `0.258174`, dev B는
`0.254591`이었습니다. dev A가 aggregate Dice와 boundary F1에서 가장 높았지만
v0 대비 paired bootstrap 구간이 0을 포함해 확실한 개선으로 판정하지 않습니다.
현재 기준 체크포인트는 v0의 epoch 16으로 유지합니다. 이 평가는 검수 정상 데이터가
없는 개발용 결과이며 held-out test는 열지 않았습니다. 자세한 수치는
[학습 및 실험 결과](TRAINING_AND_RESULTS.md)를 참조하세요.

v0.3에서는 동일한 augmentation·loss·30 epoch 예산으로 UPerNet-Swin-Tiny,
SegFormer-B1, SegFormer-B2를 비교했습니다. validation Dice는 각각 `0.250571`,
`0.237123`, `0.244192`로 모두 v0보다 낮았고, 13개 group paired bootstrap
95% 구간도 모두 음수였습니다. SegFormer는 더 빠르고 메모리를 적게 사용했지만
현재 데이터에서는 정확도 개선이 없어 모델 교체 후보에서 제외했습니다.

## 문서 목록

1. [우선순위 실행 계획](EXECUTION_PLAN.md)
2. [실험 체크포인트 레지스트리](EXPERIMENT_REGISTRY.json)
3. [프로젝트 구조와 아키텍처](ARCHITECTURE.md)
4. [데이터와 라벨링](DATA_AND_LABELING.md)
5. [학습 및 실험 결과](TRAINING_AND_RESULTS.md)
6. [개발·학습·원격 운영 절차](OPERATIONS.md)
7. [다음 단계와 운영 준비 기준](ROADMAP.md)
8. [원본 데이터 감사 결과](DATA_AUDIT.md)

## 핵심 원칙

- 원본 외장 디스크는 읽기 전용 입력으로 취급합니다.
- 이미지가 마스크와 매칭되지 않는다고 해서 정상 이미지로 간주하지 않습니다.
- 사람이 `negative`로 확인한 패치만 정상 학습 데이터에 포함합니다.
- 동일 원본 그룹은 train/validation/test 사이에 섞지 않습니다.
- 임계값은 validation에서만 선택하고 test는 최종 평가에 한 번만 사용합니다.
- 양성 데이터만 사용한 v0와 v0.1은 개발용 모델이며 운영 판정에 사용하지 않습니다.

## 빠른 시작

```bash
uv sync --extra dev
uv run ourbrain-cv --help
uv run pytest
uv run ruff check .
```

전체 명령 흐름은 저장소 루트의 [README](../README.md)와
[운영 절차](OPERATIONS.md)를 참조하세요.
