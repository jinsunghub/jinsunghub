# Speculative Decoding Lab

간단한 코드로 **Speculative Decoding**을 실습할 수 있는 미니 레포입니다.

## 구성

- `spec_decode_toy.py`: 외부 모델 없이 동작하는 장난감(Toy) 실습 코드
- `spec_decode_hf.py`: **Colab GPU + Llama 7B** 기준 실습 코드(베이스라인 vs speculative 비교)
- `requirements.txt`: 필요한 패키지 목록
- `speculative_decoding_colab.ipynb`: Colab에서 바로 실행 가능한 주피터 노트북

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

## Colab + GPU에서 7B 모델 실습 (권장: 공개 모델 Qwen 2.5 7B)

가장 간단한 방법은 노트북 파일(`speculative_decoding_colab.ipynb`)을 Colab에서 여는 것입니다.


> 권장: Colab 런타임을 `GPU(T4/L4/A100)`로 설정

### 1) 환경 준비

```bash
!rm -rf /content/jinsunghub
%cd /content
!git clone <YOUR_REPO_URL> jinsunghub
%cd /content/jinsunghub/speculative-decoding-lab
!pip install -U pip
!pip install -r requirements.txt
!python spec_decode_hf.py --help
!python - <<'PY'
from pathlib import Path
text = Path("spec_decode_hf.py").read_text(encoding="utf-8")
print("VERSION_OK" if "builtin-assistant-v2" in text else "OLD_SCRIPT")
PY
```

### 2) Hugging Face 로그인 (선택)

```bash
from huggingface_hub import login
login()
```

### 3) 실행

> `unrecognized arguments: --num-assistant-tokens`가 뜨면, 오래된 파일이 실행 중인 경우가 많습니다.
> 위처럼 `/content/jinsunghub`를 지우고 다시 clone한 뒤 실행하세요.

> 오래 걸리거나 멈춘 것처럼 보이면 더 작은 `--max-new-tokens`(예: 32~64)를 사용하세요.

```bash
!python spec_decode_hf.py \
  --target-model Qwen/Qwen2.5-7B-Instruct \
  --draft-model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain speculative decoding in simple Korean." \
  --max-new-tokens 64 \
  --k 8
```



### (선택) `--skip-baseline` 지원 버전에서 더 빠르게 돌리기

아래가 `True`면 `--skip-baseline`을 붙여 speculative만 실행할 수 있습니다.

```bash
!python - <<'PY'
import subprocess
help_text = subprocess.check_output(["python", "spec_decode_hf.py", "--help"], text=True)
print("--skip-baseline" in help_text)
PY
```

`True`일 때 실행 예시:

```bash
!python spec_decode_hf.py --target-model Qwen/Qwen2.5-7B-Instruct --draft-model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain speculative decoding in simple Korean." --max-new-tokens 64 --k 8 --skip-baseline
```

### (선택) Llama 2를 꼭 쓰고 싶다면

`meta-llama/Llama-2-7b-hf`는 gated repo라서 접근 승인 없으면 403 오류가 납니다.

1. https://huggingface.co/meta-llama/Llama-2-7b-hf 에서 접근 요청/승인
2. Colab에서 `login()`으로 승인된 계정 토큰 로그인
3. 아래 명령으로 실행

```bash
!python spec_decode_hf.py \
  --target-model meta-llama/Llama-2-7b-hf \
  --draft-model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --prompt "Explain speculative decoding in simple Korean." \
  --max-new-tokens 64 \
  --k 8
```

## 핵심 개념 정리

1. 작은 드래프트 모델이 여러 토큰을 먼저 제안
2. 큰 타깃 모델이 이를 검증/보정
3. 승인율이 높을수록 타깃 연산을 줄여 속도 향상을 기대

## 참고

- 최신 버전은 `--num-assistant-tokens`를 사용하고, 이전 방식 `--k`도 호환됩니다.
- `spec_decode_hf.py`는 `transformers`의 `assistant_model` 기반 speculative decoding 경로를 사용합니다.
- 모델/토크나이저 호환성에 따라 성능이나 동작이 달라질 수 있습니다.
