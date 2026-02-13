# Speculative Decoding Lab

간단한 코드로 **Speculative Decoding**을 실습할 수 있는 미니 레포입니다.

## 구성

- `spec_decode_toy.py`: 외부 모델 없이 동작하는 장난감(Toy) 실습 코드
- `spec_decode_hf.py`: Hugging Face 모델 2개(드래프트/타깃)로 실습하는 코드
- `requirements.txt`: 필요한 패키지 목록

## 빠른 시작

```bash
cd speculative-decoding-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python spec_decode_toy.py --prompt "I" --max-new-tokens 20 --k 4 --seed 7
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

## Hugging Face 실습(선택)

> GPU가 없으면 느릴 수 있습니다.

```bash
python spec_decode_hf.py \
  --target-model gpt2 \
  --draft-model sshleifer/tiny-gpt2 \
  --prompt "The future of AI is" \
  --max-new-tokens 40 \
  --k 4
```

## 핵심 개념 정리

1. 작은 드래프트 모델이 `k`개 토큰을 먼저 제안
2. 큰 타깃 모델이 왼쪽부터 순서대로 검증
3. 불일치 지점이 나오면 그 지점의 타깃 토큰으로 보정 후 다시 반복
4. 승인 비율이 높을수록 타깃 모델 호출 횟수를 줄여 더 빠를 가능성이 큼

## 실습 아이디어

- `k`를 1~8 범위로 바꿔 acceptance rate와 처리 시간 비교
- 드래프트 모델 크기를 키워 acceptance rate가 얼마나 변하는지 확인
- 동일 프롬프트에서 greedy vs speculative 출력을 비교
