# 다음 단계와 운영 준비 기준

## 현재 결론

코드와 실행 인프라는 개발 모델을 학습하고 대형 BMP를 추론할 수 있는 수준입니다.
그러나 현재 데이터는 양성 마스크 위주이므로 `균열 없음` 판정의 오탐률을 증명할 수
없습니다. v0.1과 v0.3의 UPerNet/SegFormer 후보도 v0의 주 지표와 통계 gate를
넘지 못했습니다. 현재 상태를 최종 모델로 포장하면 안 되며, 다음 성능 개선의
우선순위는 추가 아키텍처 탐색보다 정상/hard-negative 검수입니다.

## 단계 1: 정상 후보 200장 검수

현재 1장만 검수됐고 199장이 남았습니다.

완료 조건:

- 200/200 결정
- unresolved conflict 0
- train/validation/test에 `negative`가 각각 존재
- 터널 구조물과 오염을 포함한 hard-negative 확보
- CSV와 review audit의 SHA-256 일치

## 단계 2: 최종 학습용 manifest 생성

```bash
uv run ourbrain-cv remote-review-download \
  --url https://ourbrain-tunnel-review.vercel.app \
  --output data/negative_review/negative_review_reviewed.csv

uv run ourbrain-cv import-negatives \
  --review data/negative_review/negative_review_reviewed.csv \
  --manifest artifacts/manifest.csv \
  --output artifacts/manifest_with_negatives.csv
```

완료 조건:

- strict import 성공
- review audit 검증 성공
- split별 양성/정상 개수 문서화
- group leakage 0

## 단계 3: v0.2 통제 실험

같은 데이터, seed, split과 학습 예산을 사용해 다음 두 실험을 비교합니다.

| 실험 | 목적 |
|---|---|
| A: v0 objective + reviewed negatives | 안정적인 기준선 |
| B: v0.1 sampling/loss + reviewed negatives | recall·중심선 연결성 개선 검증 |

완료 조건:

- 두 실험 모두 정상 종료
- best/last checkpoint 보존
- config, source commit, manifest hash 기록
- validation 지표와 실패 사례 비교

v0.1이 양성 데이터에서 Dice를 높이지 못했으므로 B를 자동 승자로 간주하지 않습니다.

## 단계 4: 임계값 보정과 최종 test

1. validation에서 이미지 recall 하한 0.95를 적용
2. 조건을 만족하는 후보 중 specificity 최대 임계값 선택
3. 임계값과 후처리 설정 고정
4. held-out test를 한 번 평가

필수 보고 지표:

- image-level recall
- image-level specificity
- crack Dice / IoU
- pixel precision / recall
- boundary F1
- false positive / false negative 사례
- 구조물 유형별 오류 분류

specificity의 최종 합격 기준은 OurBrain과 현장 담당자가 운영 비용을 기준으로
확정해야 합니다.

## 단계 5: 대형 BMP 파일럿

held-out test를 통과한 모델만 실제 대형 BMP에 적용합니다.

검증 항목:

- 전체 이미지 처리 시간과 최대 메모리
- 타일 경계 artifact
- 케이블, 이음부, 오염, 조명 반사의 오탐
- 매우 얇고 흐린 균열의 누락
- `maximum_positive_ratio` 품질 게이트 동작
- overlay를 이용한 현장 전문가 확인

## 단계 6: 운영 패키지

운영 승인 시 다음 산출물을 고정합니다.

- 모델 checkpoint와 SHA-256
- config
- threshold calibration JSON
- manifest와 review audit
- test metrics
- Python package/wheel
- 모델 카드와 사용 제한
- 입력/출력 스키마
- 배포 환경의 smoke test
- rollback 가능한 이전 모델

Vercel은 검수 UI에만 사용하고, 실제 GPU 추론은 GPU 서버·워크스테이션 또는 별도의
추론 서비스에 배치합니다.

## 운영 승인 전 금지 사항

- v0/v0.1을 최종 정확도 모델로 홍보
- 양성-only validation의 이미지 specificity `1.0` 인용
- test split으로 threshold 튜닝
- 라벨 없는 이미지를 자동 정상 처리
- 검수 감사 파일을 우회한 수동 CSV 병합
- 임계값이나 후처리를 바꾼 뒤 기존 test 수치를 재사용
