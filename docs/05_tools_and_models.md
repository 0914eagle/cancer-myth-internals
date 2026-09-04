# 05. 도구와 모델

## 공개 체크포인트 — Gemma·Llama 전부

| 모델 | NLA (kitft) | AO (adamkarvonen) | Cancer-Myth Table 3 | SAE 스위트 |
|---|---|---|---|---|
| **Gemma-3-27B-IT** | ✅ L41 | ✅ | ✗ (Plain 직접 측정) | 확인 필요 |
| Gemma-3-12B-IT | ✅ L32 | ✗ | ✗ | 확인 필요 |
| **Llama-3.3-70B-Instruct** | ✅ L53 | ✅ | ✗ | ✗ |
| **Gemma-2-27B-it** | ✗ | ✅ | ✅ **17.3 (1위)** | Gemma Scope ✅ |
| Gemma-2-9B-it | ✗ | ✅ | ✗ | Gemma Scope ✅ |
| **Llama-3.1-8B-Instruct** | ✗ | ✅ | ✅ 4.8 | Llama Scope ✅ 전 층 256개 |
| Llama-3.1-70B | ✗ | ✗ | ✅ 6.3 | ✗ |
| Qwen2.5-7B-Instruct | ✅ L20 | ✗ | ✅ 6.3 | ✗ (공식 없음) |
| Qwen3-8B | ✗ | ✅ | ✗ | Qwen-Scope ✅ |

NLA 체크포인트 (HF `kitft/`): `nla-qwen2.5-7b-L20-{av,ar}`, `nla-gemma3-12b-L32-{av,ar}`, `nla-gemma3-27b-L41-{av,ar}`, `Llama-3.3-70B-NLA-L53-{av,ar}`.
AO 체크포인트 (HF `adamkarvonen/` 컬렉션 12개): gemma-2-9b-it, gemma-2-27b-it, gemma-3-1b-it, gemma-3-27b-it, Qwen3-1.7B/4B/8B/14B/32B, Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct. (1~4B는 OOD 성능 약하다고 저자 주의)

## 배치

| 모델 | 역할 | 이유 |
|---|---|---|
| **Gemma-3-27B-IT** | verbalizer 비교의 본진 | NLA + AO 둘 다 있음 → 같은 활성값에 두 verbalizer. Plain PCR은 `evaluate.py`로 한 번 측정해 Table 3 새 행으로 |
| **Gemma-2-27B-it** | 개입의 본진 | Table 3 1위 행을 직접 개선. Gemma Scope로 feature 수준 개입 가능 |
| **Llama-3.1-8B-Instruct** | sweep·probe | 싸고(24GB), Llama Scope 전 층, Table 3 행 |
| **Llama-3.3-70B** | 70B 재현 | NLA + AO. bf16 ~140GB. 컴퓨트 되면 |
| Qwen2.5-7B-Instruct | NLA 판독 보조 | Table 3 행이고 ②의 Table 8 baseline 8종이 같은 데이터로 있음 |

## 메모리

- Gemma-2-27B / Gemma-3-27B bf16 ≈ 54GB → A100/H100 80GB 1장
- Llama-3.1-8B bf16 ≈ 16GB → 24GB
- Llama-3.3-70B bf16 ≈ 140GB → 80GB 2장
- verbalizer(AV/AO)는 타깃과 같은 크기 → 별도 서빙 (SGLang)

## 저장소·코드

| | |
|---|---|
| Cancer-Myth | [Bill1235813/cancer-myth](https://github.com/bill1235813/cancer-myth) — `evaluate.py`(dspy), `validate.py`, `validate_nfp.py`, `data/all_data.json`, `data/nfp.json` |
| ② Well | [ShenranTomWang/Well](https://github.com/ShenranTomWang/Well) — FPQ/TPQ 분할, 6개 방법군, GEPA·FAITH·LoRA 스크립트, evaluator 8종. 지원 계열 `llama3.2, qwen2.5, qwen3, gemma4, llama3, olmo3` |
| NLA | [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders) (학습, 938★), [kitft/nla-inference](https://github.com/kitft/nla-inference) (추론), [asherps/EasyNLA](https://github.com/asherps/EasyNLA), [ceselder/nanoNLA](https://github.com/ceselder/nanoNLA) |
| AO | [adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles) — 데모 노트북, 학습 스크립트 |
| SAE | Gemma Scope (Gemma-2 2B/9B/27B), [fnlp/Llama-Scope](https://huggingface.co/fnlp/Llama-Scope) (Llama-3.1-8B, 256 SAEs), Qwen-Scope (`Qwen/SAE-Res-Qwen3-*`) |
| 행동 방향 | CAA ([2312.06681](https://arxiv.org/abs/2312.06681)), Refusal direction (Arditi et al. 2024), Persona vectors (Anthropic 2025), SAS ([2503.00177](https://arxiv.org/html/2503.00177)) |
| Negative bias | NAS/NASA ([2408.00137](https://arxiv.org/abs/2408.00137), NAACL 2025), [2511.10881](https://arxiv.org/abs/2511.10881) |
