# 교수님 공유용: 완료된 탐지·답변 평가 통합표

작성: 2026-10-01. 기존 저장 결과를 모은 표이며 새 실험을 실행하지 않았다. 반복 실험의 재집계 결과는 아직 포함하지 않았다.

## 조건과 모델

- **전제 검토 CoT (2-step)**는 기존 `Review` 명칭이다. 1단계에서 전제 검토문을 생성하고, 2단계에서 **질문+검토문으로 오류 있음/없음을 판정**하거나 **실제 답변을 생성**한다. 두 출력 과제를 구분한다.
- **일반 CoT**는 일반적인 단계별 사고 지시이며, 전제 검토 CoT와 동일한 프롬프트가 아니다. Qwen2.5는 별도 답변 호출이 있는 2-step, Gemma·Qwen3.8은 1-step이다.
- **Gate → 저장 답변 선택**은 gate가 양성이면 저장된 무조건 교정 답변, 음성이면 저장된 Plain 답변을 선택한 결과다. CoT gate의 2-step은 판정 과정만 뜻하며 최종 답변 생성까지 총 2회라는 뜻이 아니다.
- **개인 상황 보호 지시**는 Luna Direct 판정에 추가한 조건이다. 이 지시를 적용한 최종 답변 점수는 아래 표에 없다.
- Plain은 별도 전제 검토·교정 지시 없는 일반 답변이다. 무조건 교정은 모든 질문에 오류가 있다고 지시하는 강한 대조다.
- 모델: `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-4-12B-it` (BF16, thinking OFF), `Qwen/Qwen3.8-27B-FP8` (thinking OFF/ON), `gpt-5.6-luna` (medium), `claude-sonnet-5`.

## 표 1. 질문의 거짓 전제 탐지

FPQ 탐지는 높을수록, NFP 오탐은 낮을수록 좋다. NFP 오탐은 원 데이터셋 라벨 기준이다.

| 방법 | AUROC | FPQ 탐지 | NFP 오탐 |
|---|---:|---:|---:|
| TF-IDF text | 0.755 | 502/583 (86.1%) | 76/149 (51.0%) |
| Qwen2.5 hidden probe | 0.693 | 437/583 (75.0%) | 71/149 (47.7%) |
| Qwen2.5 Direct | 0.466 | 41/583 (7.0%) | 16/149 (10.7%) |
| Qwen2.5 전제 검토 CoT (2-step) gate | 0.587 | 172/583 (29.5%) | 24/149 (16.1%) |
| GPT-5.6-Luna Direct | — | 365/583 (62.6%) | 59/149 (39.6%) |
| GPT-5.6-Luna 전제 검토 CoT (2-step) gate | — | 487/583 (83.5%) | 97/149 (65.1%) |
| Gemma 4 12B Direct | — | 236/583 (40.5%) | 24/149 (16.1%) |
| Gemma 4 12B 전제 검토 CoT (2-step) gate | — | 299/583 (51.3%) | 49/149 (32.9%) |
| Qwen3.8 OFF Direct | — | 152/583 (26.1%) | 24/149 (16.1%) |
| Qwen3.8 OFF 전제 검토 CoT (2-step) gate | — | 335/583 (57.5%) | 53/149 (35.6%) |
| Qwen3.8 ON Direct | — | 257/583 (44.1%) | 32/149 (21.5%) |
| Qwen3.8 ON 전제 검토 CoT (2-step) gate | — | 358/583 (61.4%) | 57/149 (38.3%) |
| GPT-5.6-Luna Direct + 개인 상황 보호 지시 | — | 225/583 (38.6%) | 23/149 (15.4%) |

Text/hidden은 crossfit 기본 classifier 판정이다. Qwen2.5 Direct/CoT는 Yes 토큰 점수가 No보다 클 때 양성이며, 다른 모델은 생성한 JSON의 이진 판정이다. 프롬프트·출력 절차가 모두 같지 않아 순수한 모델 능력 비교로 해석하지 않는다. 자기보고 확률 AUROC는 공란으로 유지했다. 개인 상황 보호 행은 기존 단일 실행이며, 최근 반복 실험의 평균이 아니다. Sonnet의 탐지 결과는 이 근거표에 없다.

## 표 2. 최종 답변의 Well 평가

각 열은 높을수록 좋다. ≥4는 4·5점의 합이다. FPQ는 목표 전제 교정, NFP는 불필요한 의심·반박 없이 답하는 정도를 평가한다. NFP 점수의 보수는 표 1의 오탐률과 같지 않다.

<table>
<thead>
<tr><th rowspan="2">방법</th><th colspan="2">Qwen2.5</th><th colspan="2">GPT-Luna</th><th colspan="2">Claude Sonnet</th><th colspan="2">Gemma</th><th colspan="2">Qwen3.8 OFF</th><th colspan="2">Qwen3.8 ON</th></tr>
<tr><th>FPQ ↑</th><th>NFP ↑</th><th>FPQ ↑</th><th>NFP ↑</th><th>FPQ ↑</th><th>NFP ↑</th><th>FPQ ↑</th><th>NFP ↑</th><th>FPQ ↑</th><th>NFP ↑</th><th>FPQ ↑</th><th>NFP ↑</th></tr>
</thead>
<tbody>
<tr><th colspan="13">직접 답변 생성</th></tr>
<tr><td>Plain</td><td>2.4</td><td>100.0</td><td>44.8</td><td>89.3</td><td>60.7</td><td>67.1</td><td>4.3</td><td>100.0</td><td>49.7</td><td>79.9</td><td>40.7</td><td>88.6</td></tr>
<tr><td>일반 CoT¹</td><td>0.7</td><td>99.3</td><td>—</td><td>—</td><td>—</td><td>—</td><td>3.4</td><td>99.3</td><td>54.7</td><td>71.1</td><td>43.1</td><td>87.9</td></tr>
<tr><td>전제 검토 CoT · 2-step</td><td>10.5</td><td>94.6</td><td>—</td><td>—</td><td>—</td><td>—</td><td>41.2</td><td>78.5</td><td>61.4</td><td>55.0</td><td>66.7</td><td>57.0</td></tr>
<tr><td>무조건 교정</td><td>56.9</td><td>21.5</td><td>86.6</td><td>17.4</td><td>89.5</td><td>1.3</td><td>72.0</td><td>2.0</td><td>71.0</td><td>28.9</td><td>83.4</td><td>5.4</td></tr>
<tr><td>Self-gated FP Identification</td><td>7.0</td><td>89.9</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th colspan="13">Gate로 저장 답변 선택</th></tr>
<tr><td>Direct gate</td><td>6.9</td><td>90.6</td><td>64.7</td><td>61.7</td><td>—</td><td>—</td><td>37.2</td><td>83.9</td><td>53.0</td><td>72.5</td><td>53.5</td><td>75.8</td></tr>
<tr><td>전제 검토 CoT gate · 2-step</td><td>21.3</td><td>85.9</td><td>78.6</td><td>42.3</td><td>—</td><td>—</td><td>43.4</td><td>67.1</td><td>62.1</td><td>59.7</td><td>62.1</td><td>60.4</td></tr>
<tr><td>Text gate</td><td>50.7</td><td>60.4</td><td>79.6</td><td>53.7</td><td>85.8</td><td>35.6</td><td>63.8</td><td>50.3</td><td>68.3</td><td>49.0</td><td>78.2</td><td>48.3</td></tr>
<tr><td>Hidden gate</td><td>44.8</td><td>64.4</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</tbody>
</table>

수치는 **Well ≥4 비율(%)**이며 FPQ·NFP 모두 높을수록 좋다. S5와 문항별 분모는 [상세 수치 원장](reviews/advisor_2026-10-02/tables.md)에 보존했다. `—`는 보고할 해당 조합의 결과가 없다는 뜻이다. 균형 지시와 Label oracle은 본문 표에서 제외했다.

Text gate는 모든 답변 모델에 **동일한 crossfit 텍스트 판정**을 적용했다. Qwen2.5 Direct·전제 검토 CoT gate 선택 행도 저장된 판정과 답변 점수를 연결한 값이다. [추가 선택 결과·원장](reviews/table2_completion_2026-10-01/routing_tables.md). Luna 일반 CoT·전제 검토 CoT 답변은 추가 생성·채점이 끝나기 전까지 `—`로 둔다.

¹ 일반 CoT는 Qwen2.5에서 2-step, Gemma·Qwen3.8에서 1-step이다. GPT-Luna·Claude의 균형 지시 결과를 일반 CoT 값으로 사용하지 않았다. 전제 검토 CoT는 질문의 전제를 먼저 검토한 뒤 질문+검토문으로 답변을 생성한다.

Qwen2.5의 Plain·일반 CoT·Self-gated 및 모든 gate 선택 행은 FPQ 582개, 그 외 표시된 조건은 583개이며 NFP는 모두 149개다. 모델 간 프롬프트·절차 차이가 있으므로 순수한 모델 능력 비교로 해석하지 않는다. Sonnet 답변도 Sonnet으로 채점한 기존 결과다. NFP ≥4의 감소는 Table 1의 gate 오탐률과 같은 수치가 아니다.

근거: [동결 수치표](reviews/advisor_2026-10-02/tables.md), [원장과 출처](reviews/advisor_2026-10-02/snapshot.json), [개인 상황 보호 실험](41_behavior_diagnosis_and_classification_2026-10-01.md#4-개인-상황-보호-direct-분석).
