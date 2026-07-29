# OurBrain 데이터 감사 결과

검사일: 2026-07-29  
입력: `/Volumes/새 볼륨`  
정책: 원본 읽기 전용

## 학습 페어

| 항목 | 수량 |
|---|---:|
| 패치 이미지 | 1,886 |
| 라벨 파일 | 1,237 |
| 이름이 매칭된 페어 | 1,224 |
| 정상 디코딩 페어 | 1,223 |
| 이미지에 라벨 없음 | 662 |
| 이미지 없는 라벨 | 13 |
| 유효 source group | 89 |

## 그룹 분할

동일한 첫 번째 filename prefix가 여러 split에 들어가지 않도록 분할했습니다.

| split | 그룹 | 패치 |
|---|---:|---:|
| train | 62 | 774 |
| validation | 13 | 221 |
| test | 14 | 228 |

그룹 누수 검사 결과: **0개**

## 마스크

| 원본 마스크 크기 | 수량 |
|---|---:|
| 512×512 | 589 |
| 682×682 | 633 |
| 711×711 | 1 |

- 마스크의 검정 픽셀을 `crack=1`로 사용합니다.
- 이미지 크기와 다른 마스크는 categorical label 보존을 위해
  nearest-neighbor로 512×512에 맞춥니다.
- 유효 마스크의 crack 픽셀은 51~2,133개이며 전체 710,984개입니다.
- 라벨은 균열 폭 영역보다 중심선에 가깝기 때문에 crack width 계측용으로는
  충분하지 않습니다.

## 손상 파일

다음 페어는 마스크 디코딩 오류로 manifest에서 제외했습니다.

```text
stem: 0348_019_008
mask: /Volumes/새 볼륨/train/label/crack/0348_019_008-L.bmp
error: image file is truncated (1994 bytes not processed)
```

## 생성물

- `artifacts/manifest.csv`: 유효 1,223개 페어
- `artifacts/data_audit.json`: 전체 missing/orphan/손상 상세
- `data/negative_review/`: 전체 원본에서 표본화한 정상 검수 후보 200개
- `data/negative_review/contact_sheet_000.jpg`~`012.jpg`: 검수용 모음 13장
- `data/negative_review/negative_review.csv`: 아직 라벨이 비어 있는 검수 입력
- `data/negative_review/review.html`: 단축키·진행 저장·CSV export를 지원하는 로컬 UI
- `data/negative_review_smoke/`: 첫 원본 BMP에서 생성한 정상 검수 후보 16개
- `data/negative_review_smoke/contact_sheet_000.jpg`: 검수용 모음

전체 검수 후보는 원본 스캔 86개에서 추출되었고, 기존 group split 기준 예상
분포는 train 136개, validation 38개, test 26개입니다. 원본 BMP는 수정하지
않았으며, 비압축 24-bit BMP의 필요한 512×512 행 범위만 읽어 패치를 만들었습니다.

## 학습 전 남은 필수 작업

`균열 없음` 성능을 증명하려면 정상 후보의 `review_label`을 터널 검사 도메인
담당자가 검수해야 합니다. 이음부, 타일/그리드, 케이블, 조명, 오염, 누수 흔적을
포함한 hard-negative가 test split에 반드시 포함돼야 합니다.

기본 학습 명령은 `reviewed_negative`가 train/validation/test에 각각 한 개 이상
없으면 중단됩니다. `--allow-positive-only`는 실행 경로 스모크 테스트 전용이며,
그 옵션으로 만든 체크포인트는 성능 판정이나 배포에 사용할 수 없습니다.
