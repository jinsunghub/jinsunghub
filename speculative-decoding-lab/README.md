# Speculative Decoding Lab

간단한 코드로 **Speculative Decoding**을 실습할 수 있는 미니 레포입니다.

## 구성

- `spec_decode_toy.py`: 외부 모델 없이 동작하는 장난감(Toy) 실습 코드
- `spec_decode_hf.py`: **Colab GPU + Llama 7B** 기준 실습 코드(베이스라인 vs speculative 비교)
- `requirements.txt`: 필요한 패키지 목록

## 빠른 시작 (Toy)

### macOS / Linux (bash, zsh)

```bash
cd speculative-decoding-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python spec_decode_toy.py --prompt "I" --max-new-tokens 20 --k 4 --seed 7
```

### Windows PowerShell

```powershell
cd speculative-decoding-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\spec_decode_toy.py --prompt "I" --max-new-tokens 20 --k 4 --seed 7
```

> `source .venv/bin/activate`는 Linux/macOS 셸 명령이라 PowerShell에서는 동작하지 않습니다.

### Windows cmd

```bat
cd speculative-decoding-lab
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python spec_decode_toy.py --prompt "I" --max-new-tokens 20 --k 4 --seed 7
```

### PowerShell에서 실행 정책 오류가 날 때

`Activate.ps1` 실행 시 정책 오류가 나면, 현재 사용자 범위로 한 번만 설정하세요.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

그 다음 다시 활성화:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Toy 실습 포인트

- 드래프트 모델이 제안한 토큰을 타깃 모델이 얼마나 자주 승인(accept)하는지 확인
- `k`(한 번에 제안하는 토큰 수) 값을 바꿔 acceptance rate 비교
- 드래프트/타깃 모델 분포 차이를 바꿔 속도-품질 트레이드오프 관찰

예시:

```bash
python spec_decode_toy.py --prompt "I" --max-new-tokens 30 --k 2
python spec_decode_toy.py --prompt "I" --max-new-tokens 30 --k 6
```

## Colab + GPU에서 Llama 7B 실습

> 권장: Colab 런타임을 `GPU(T4/L4/A100)`로 설정

### 1) 환경 준비

```bash
!git clone <YOUR_REPO_URL>
%cd speculative-decoding-lab
!pip install -U pip
!pip install -r requirements.txt
```

### 2) Hugging Face 로그인 (Llama 2 접근권한 필요)

```bash
from huggingface_hub import login
login()
```

### 3) 실행

```bash
!python spec_decode_hf.py \
  --target-model meta-llama/Llama-2-7b-hf \
  --draft-model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --prompt "Explain speculative decoding in simple Korean." \
  --max-new-tokens 128 \
  --num-assistant-tokens 8
```

## 핵심 개념 정리

1. 작은 드래프트 모델이 여러 토큰을 먼저 제안
2. 큰 타깃 모델이 이를 검증/보정
3. 승인율이 높을수록 타깃 연산을 줄여 속도 향상을 기대

## 참고

- `spec_decode_hf.py`는 `transformers`의 `assistant_model` 기반 speculative decoding 경로를 사용합니다.
- 모델/토크나이저 호환성에 따라 성능이나 동작이 달라질 수 있습니다.
