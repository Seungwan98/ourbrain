# Third-Party Model and Software Notices

이 저장소의 자체 코드는 별도 배포 계약에 따릅니다. 외부 모델 및 라이브러리의
원 라이선스는 각각 유지됩니다.

## Baseline checkpoint

- `openmmlab/upernet-swin-tiny`
  - Hugging Face model card license: MIT
  - https://huggingface.co/openmmlab/upernet-swin-tiny
  - UPerNet and Swin Transformer pretrained weights are used as a fine-tuning
    initialization. Distribution must retain applicable notices.

## Runtime libraries

- Hugging Face Transformers: Apache-2.0
- PyTorch / TorchVision: BSD-style licenses
- Pillow: HPND
- NumPy: BSD-3-Clause
- PyYAML: MIT
- safetensors: Apache-2.0
- Vercel Blob SDK (`@vercel/blob`): Apache-2.0

배포 시 실제 lockfile 버전의 라이선스 텍스트를 다시 수집하고, 고객사 계약과
모델 배포 방식에 맞는 법률 검토를 수행해야 합니다.
