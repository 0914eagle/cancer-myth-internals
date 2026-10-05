# Classic false-premise / questionable-assumption QA (2021–2024): the FPQ-vs-NFP trade-off

**How these notes were sourced (read first).** Direct fetch of arxiv.org, ar5iv, aclanthology, alphaxiv, huggingface, semanticscholar, openreview, papers.cool, liner, emergentmind, themoonlight, pith.science and lacuna was **blocked** here, for WebFetch and for curl through the proxy. Only github.com pages could be fetched. So almost every claim below comes from **WebSearch result snippets and the search tool's summaries of the paper PDFs/HTML**, not from my own full-text reading. The search summaries sometimes merge text from several papers. I flag the cases where I saw that happen. Numbers marked "(snippet)" should be checked against the PDF before they are quoted in a final report. Labels used:
- **[explicit]**: the paper (per snippet) states it.
- **[implied]**: follows from the paper's numbers or setup.
- **[unverified]**: from memory, or the attribution is uncertain.

---

## Q1. Kim et al. 2021 ("Which Linguist Invented the Lightbulb?") and Kim et al. 2023 ((QA)²): do they report false-positive flags on valid questions, and do they analyze why?

### Takeaway
Neither paper frames over-flagging of valid questions as a side effect of improving false-premise detection, and neither gives a mechanism for it. Kim 2021 is an NQ-unanswerability study. Its end-to-end gain was "modest", and it does not, as far as I could verify, report a separate false-flag rate on answerable questions. (QA)² is the first in this line to build the evaluation so that valid-assumption questions are scored too. That design is what later lets response bias (always "yes" or always "no") show up. The bias analysis I could verify, though, sits mostly in the follow-up Syn-(QA)² (2024), not in (QA)² itself.

### Cited Findings
**Kim, Pavlick, Karagol Ayan, Ramachandran (ACL 2021)**
- (a) Domain: open-domain QA on Natural Questions (NQ). About 21% of NQ's unanswerable questions can be explained by unverifiable presuppositions. [explicit] — [ACL Anthology](https://aclanthology.org/2021.acl-long.304/); [arXiv 2101.00391](https://arxiv.org/abs/2101.00391)
- (b) Method: a three-step framework: presupposition generation, then presupposition verification, then explanation generation. [explicit] — [arXiv PDF](https://arxiv.org/pdf/2101.00391)
- Result: "adding presuppositions and their verifiability to an existing model yields modest gains in downstream performance and unanswerability detection." [explicit, snippet] — [arXiv 2101.00391](https://arxiv.org/abs/2101.00391)
- A user preference study found that the *oracle* behavior of the presupposition-failure-based system is preferred over the oracle behavior of existing QA systems. [explicit, as paraphrased by a later paper] — [Wang, Shwartz & Gonen 2026, arXiv 2608.06539](https://arxiv.org/html/2608.06539)
- Later work describes Kim 2021 as the template for the "FPQA pipeline": (a) extract atomic presuppositions, (b) fact-check each one, usually against source documents, (c) generate a response from the fact-check results. That 2026 paper argues that evaluation in this line "largely focuses on" false-presupposition questions (FPQs) "while ignoring performance on 'normal' questions (TPQs)." [explicit in the 2026 paper] — [arXiv 2608.06539](https://arxiv.org/html/2608.06539)
- (c)/(d): I found no snippet in which Kim 2021 reports a false-positive presupposition-failure rate on answerable NQ questions, or a loss on answerable questions. NQ includes answerable questions, so the answerability-detection metric implicitly scores them. [implied / not verified]

**Kim, Htut, Bowman, Petty ((QA)², ACL 2023)**
- (a) Domain: open-domain evaluation set of naturally occurring search-engine (Google autocomplete) queries that "may or may not contain questionable assumptions." Success requires producing "adequate responses for both typical information-seeking questions and ones with questionable assumptions." [explicit] — [ACL Anthology](https://aclanthology.org/2023.acl-long.472/); [GitHub najoungkim/QAQA](https://github.com/najoungkim/QAQA)
- (c): **Yes.** Each item carries an `all_assumptions_valid` field with values `has_invalid` / `all_valid`. Valid-assumption questions are part of the eval set. [explicit, README] — [GitHub QAQA](https://github.com/najoungkim/QAQA)
- (b)/(d) Tasks and results: abstractive QA (human-judged), binary questionable-assumption detection, and assumption verification. Best models reached about 56% abstractive-QA acceptability, 64% on detection and 72% on verification. Most models, including few-shot T-Few, were near chance on detection. [snippet; the 56/64/72 figures come from the search tool's summary and should be checked against the abstract] — [ACL PDF](https://aclanthology.org/2023.acl-long.472.pdf)
- Detection/verification asymmetry: Flan-PaLM 540B reached ~72% verifying an assumption stated in isolation, but only ~50–60% detecting the same flaw inside a question. [snippet; attribution to (QA)² is likely but not certain] — [ACL PDF](https://aclanthology.org/2023.acl-long.472.pdf)
- Response bias: Flan-T5 "heavily biased towards answering 'No'", PaLM-2 less so, and "Llama-2 showed a heavy false assumption bias, most of the times answering that a question had a false assumption regardless of the actual question content." **Attribution caution:** PaLM-2 and Llama-2 postdate (QA)²'s experiments, so this almost certainly comes from Syn-(QA)² (2024, same group). See Q5. [snippet, merged sources] — [Syn-QA2 arXiv 2403.12145](https://arxiv.org/html/2403.12145)

### Inferences
- (QA)²'s balanced has_invalid/all_valid design means a model that over-flags gets penalized on detection. A Llama-2-style "always false-assumption" bias (if it is from Syn-QA2) is exactly the over-correction failure mode. In this literature it is described as a **response bias**, not explained mechanistically. The detection-vs-verification gap (72% vs ~50–60%) hints at a cause one level down: models can verify a claim stated alone but fail to *locate* the presupposition inside a question. That is evidence about detection difficulty, not specifically about false positives.
- Neither Kim paper offers internal/causal analysis (ablations of the over-flagging, probing) of why valid questions get flagged.

### Gaps
- Could not read the (QA)² full text to confirm per-class (valid vs invalid) detection numbers or which models showed a "yes" or "no" bias in the 2023 paper itself.
- Could not confirm whether Kim 2021 reports precision/false-positive rates for its verifiers on answerable NQ questions, or what error analysis it gives. Full text was blocked.

---

## Q2. CREPE (Yu, Min, Zettlemoyer, Hajishirzi, ACL 2023): baselines on non-false-presupposition questions; any error analysis of false detections?

### Takeaway
CREPE uses a natural distribution: ~74% of questions are normal and ~25–26% contain a false presupposition (FP). It scores detection with **macro-F1**, so performance on normal questions counts. The error analysis I could see blames **retrieval/evidence** and the difficulty of judging factuality. I found no specific analysis of false detections on normal questions, and no mechanism for them.

### Cited Findings
- (a) 8,400 Reddit (ELI5-style) questions annotated for whether a false presupposition is present, plus the presupposition and its correction. 25% contain FPs. [explicit] — [ACL Anthology](https://aclanthology.org/2023.acl-long.583/); [ResearchGate](https://www.researchgate.net/publication/372916261_CREPE_Open-Domain_Question_Answering_with_False_Presuppositions)
- Split stats from the README: train 3,462 questions (26.2% FP), dev 2,000 (27.2%), test 3,004 (25.0%); 8,446 labeled in total (26.0% FP), plus 196,385 unlabeled examples. [explicit] — [GitHub velocityCavalry/CREPE](https://github.com/velocitycavalry/crepe)
- (b) Tasks: a *detection* subtask (is there an FP?), scored by macro-F1, and a *writing* subtask (generate the presupposition and its correction). Two tracks: question-only (main track) and question plus gold comment. The main-track baseline is a c-REALM retriever plus a multi-passage classifier. [explicit] — [arXiv HTML v1](https://arxiv.org/html/2211.17257v1); [ACL PDF](https://aclanthology.org/2023.acl-long.583.pdf)
- (d) Results: "adaptations of existing open-domain QA models can find presuppositions moderately well, but struggle when predicting whether a presupposition is factually correct." Snippets give the best main-track detection as ~67.1 macro-F1, with human ~71 F1 per one summary or "10% below human" per another. **These conflict; check the PDF.** [snippet] — [ACL PDF](https://aclanthology.org/2023.acl-long.583.pdf)
- Error analysis (as summarized): "difficulty in retrieving relevant evidence is a major factor limiting model performance on the detection subtask." Correction is much better in the gold-comment track, while presupposition identification is similar across tracks. [snippet] — [ACL PDF](https://aclanthology.org/2023.acl-long.583.pdf)

### Inferences
- Because macro-F1 covers both classes on a ~74%-normal distribution, CREPE implicitly measures over-flagging, but it does not report "false-rejection rate on normal questions" as its own quantity (not verified). The retrieval-bottleneck explanation is a cause for *weak detection in general*. The 2026 Wang et al. paper later turns this into a cause for over-correction: weak verification rejects true presuppositions it cannot verify.

### Gaps
- Per-class precision/recall, confusion matrices and the exact human F1 could not be confirmed. I could not see whether the error analysis covers false positives on normal questions specifically.

---

## Q3. FalseQA (Hu et al., ACL 2023, "Won't Get Fooled Again"): training on FPQ data harms normal questions, fixed with replay. What exactly, and did they explain why?

### Takeaway
This is the clearest **classic** report of over-correction caused by training. Fine-tuning on FalseQA makes models rebut general questions. The authors call it **"catastrophic forgetting"** and mitigate it with **data replay** of general questions (ARC-DA). With replay, the model discriminates 86.7% of FPQs and rebuts only 1.4% of general questions. The explanation stays at the level of naming the phenomenon (catastrophic forgetting). I found no evidence of internal analysis of *why* FPQ training generalizes into rebutting valid questions. This is "observe + mitigate", not mechanism.

### Cited Findings
- (a) FalseQA: 2,365 human-written FPQs, each with an explanation of the false premise and a revised true-premise question. The CSV has `label` 1 = false premise, 0 = true premise. Answers are rebuttals for FPQs and normal answers for TPQs. [explicit] — [arXiv 2307.02394](https://arxiv.org/abs/2307.02394); [GitHub thunlp/FalseQA](https://github.com/thunlp/FalseQA)
- (b) Fine-tuning pretrained LMs (pre-ChatGPT). Models can discriminate FPQs after fine-tuning on moderate amounts of data (e.g., 256 examples) and can generate reasonable rebuttal explanations. [explicit] — [ACL Anthology](https://aclanthology.org/2023.acl-long.309/)
- (c)/(d): "Training purely on FalseQA may lead to catastrophic forgetting." To keep general QA, "for each iteration over batches, a batch of data samples from general question datasets (like ARC-DA) is added", with "the same samples… kept within 30 batch iterations" to use as little general data as possible. [explicit, snippet] — [arXiv HTML](https://arxiv.org/html/2307.02394v1)
- Headline: "a simple but effective data replay method can help mitigate the catastrophic forgetting of general questions, where the model discriminates 86.7% FPQs in FalseQA and only rebuts 1.4% general questions." [explicit] — [ACL PDF](https://aclanthology.org/2023.acl-long.309.pdf)
- **Caution:** one search-tool summary misread the 1.4% as a "drop" without replay. In the paper, 1.4% is the rebuttal rate on general questions *with* replay. [my correction, based on the abstract wording above]
- README update: the authors note that ChatGPT performs "remarkably well" on FalseQA, possibly because its instruction-tuning data included false-premise questions. [explicit] — [GitHub thunlp/FalseQA](https://github.com/thunlp/FalseQA)

### Inferences
- Without replay, the general-question rebuttal rate is presumably much higher than 1.4%, since that is what "catastrophic forgetting" refers to here. I could not get the no-replay number from the snippets.
- The paper's "cause" is the generic continual-learning label (catastrophic forgetting / distribution shift toward always rebutting). That is an observation plus a mitigation, not a mechanism one level down (no probing, attention, or ablation of the over-rebuttal itself, as far as I could verify).

### Gaps
- The exact no-replay rebuttal rate on general questions, and which model sizes, could not be extracted. The full text was blocked.

---

## Q4. FAITH / "Whispers that Shake Foundations" (Yuan et al., EMNLP 2024): false-premise attention heads. Did it measure valid-premise questions or side effects?

### Takeaway
This is the main classic-period paper that gives an **internal mechanism**, but for *false-premise hallucination*, not for over-correction. Its evidence includes uncertainty analysis, information-flow analysis in shallow layers, and identification and ablation of ~1% of heads. I found **no evidence that it measured true-premise questions or general-capability side effects** of constraining the heads. One secondary summary explicitly says it does not discuss degradation of general capabilities.

### Cited Findings
- (a) Two auto-constructed datasets, Movie and Prize. Llama-2 models. [explicit] — [arXiv 2402.19103](https://arxiv.org/abs/2402.19103); [ACL Anthology](https://aclanthology.org/2024.emnlp-main.155/)
- Mechanism claims: models show more inherent uncertainty when hallucinating. Knowledge extraction about the subject is "disturbed in shallow layers." A small set of "false premise heads", mostly in shallow layers and "functioning around the false object mentioned in the question", disturbs knowledge extraction and causes the hallucination. [explicit] — [ACL PDF](https://aclanthology.org/2024.emnlp-main.155.pdf); [GitHub paper notes](https://github.com/NY1024/Foundation-Model-Paper-Notes/blob/master/llm-defense/whispers-that-shake-foundations-analyzing-and-mitigating-false-premise-hallucinations-in-large-lang.md)
- (b) FAITH constrains those heads at inference: 15 heads (0.94%) on Movie and 20 (1.25%) on Prize. This gives "nearly 20%" improvement, e.g., Llama-2-7b on Movie went from 46.53% to 77.74%. [snippet] — [ACL PDF](https://aclanthology.org/2024.emnlp-main.155.pdf)
- (c)/(d): A secondary summary says the paper "does not discuss potential degradation of general capabilities or broader performance impacts from constraining attention heads". It also says it does not detail whether true-premise questions served as a control. Method "works better on models with fewer parameters." [secondary summary] — [GitHub paper notes](https://github.com/NY1024/Foundation-Model-Paper-Notes/blob/master/llm-defense/whispers-that-shake-foundations-analyzing-and-mitigating-false-premise-hallucinations-in-large-lang.md)

### Inferences
- FAITH is the template for "mechanism of the false-premise failure." The heads are located around the false object in the question. Whether suppressing them on a *true*-premise question (where that object is correct) hurts extraction is exactly the untested side effect relevant to over-correction.

### Gaps
- Could not verify from full text whether any table reports FAITH on true-premise or general data. Treat "not measured" as likely but unconfirmed.

---

## Q5. Syn-(QA)², MultiHoax, PreWoMe, "Don't Let It Hallucinate" (2504.06438), and other methods

### Takeaway
Most benchmark papers include paired or valid questions but report detection difficulty and response bias, not a correction-induced trade-off. PreWoMe claims gains on both misleading and normal questions (no trade-off reported). The retrieval-plus-logical-form premise verifier (2025) reports an accuracy vs true-positive-rate tension across configurations, with no mechanism.

### Cited Findings
**Syn-(QA)² (Daswani, Sawant, Kim, arXiv Mar 2024)**
- (a) Two synthetic datasets, built by perturbing Wikidata relations and by perturbing HotpotQA. 1,812 question pairs (1,165 single-hop, 647 multi-hop). Each pair contrasts a false-assumption question with a counterpart that has no false assumption, so valid questions are measured by design (c). [explicit] — [arXiv 2403.12145](https://arxiv.org/html/2403.12145); [ADS](https://ui.adsabs.harvard.edu/abs/2024arXiv240312145D/abstract)
- Findings: binary detection is hard, "even compared to the difficulty of generative QA itself, possibly due to the linguistic structure of the problem", and harder on long-tail than on naturally occurring questions. [explicit] — [arXiv 2403.12145](https://arxiv.org/html/2403.12145)
- (d) Response biases (most plausibly from this paper; see the Q1 caution): Flan-T5 biased to "No". PaLM-2 less extremely biased to "No". Llama-2 "answering that a question had a false assumption regardless of the actual question content", i.e., over-flagging valid questions. Flan-T5-XXL "always answer[s] 'No'" even to trivial probes like "Is {Q} a question?" Best detection was ~62% single-hop (GPT-3.5) and ~67% multi-hop (GPT-4), 4-shot. [snippet, attribution moderately confident] — [Syn-QA2 arXiv](https://arxiv.org/html/2403.12145v1)
- (e) Explanation offered: "linguistic structure of the problem" (speculative wording, "possibly"). The trivial-probe test is a behavioral control for response bias, not an internal analysis. [explicit hedge] — [arXiv 2403.12145](https://arxiv.org/html/2403.12145)

**PreWoMe (Han et al., EMNLP 2023)**
- (a)/(b) Long-form QA. The method extracts the question's presuppositions and feeds them back as "working memory", then generates feedback and an action before the answer. Meant to handle any information-seeking question, whether ambiguous, false-presupposition or normal. [explicit] — [arXiv 2310.16147](https://arxiv.org/abs/2310.16147); [ACL Anthology](https://aclanthology.org/2023.emnlp-main.517/)
- (c)/(d): The abstract claims PreWoMe "is effective not only in tackling misleading questions but also in handling normal ones". So it does measure normal questions and reports **no** trade-off. I could not verify the numbers. [explicit claim] — [arXiv 2310.16147](https://arxiv.org/abs/2310.16147)

**"Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented Logical Reasoning" (Qin, Li, Nian, Yu, Zhao, Ma; arXiv 2504.06438, 2025, outside the core window)**
- (b) Converts the query to a logical form, retrieves evidence to verify each premise, and injects the verification result into the prompt. No logits or fine-tuning needed. [explicit] — [arXiv 2504.06438](https://arxiv.org/abs/2504.06438)
- (c)/(d): Reports accuracy and true-positive rate (TPR). When original queries (not logical forms) are used for retrieval and/or detection, a G-retriever setting gets high accuracy (91.11%) but low TPR (37.78%). "Can achieve high accuracy due to correctly identifying negatives, but is less effective at capturing false premises." So valid-premise (negative) questions are evaluated. The reported tension is toward *under*-detection in weaker configurations, not over-correction. [snippet] — [arXiv PDF](https://arxiv.org/pdf/2504.06438); [OpenReview PDF](https://openreview.net/pdf/523fd7b3e77b063a50b96da15fd6b746b26a1ffb.pdf)

**MultiHoax (Findings of ACL 2025; arXiv 2506.00264)**
- A multi-hop false-premise benchmark. 2025, outside the window; I did not investigate it. — [arXiv HTML](https://arxiv.org/html/2506.00264); [ACL PDF](https://aclanthology.org/2025.findings-acl.530.pdf)

**KG-FPQ (Zhu et al., arXiv Jul 2024 / COLING 2025)**
- ~178k KG-derived FPQs across Art, People and Place, at six confusability levels, built by editing true triplets. Finds "no evident correlation between general knowledge and resistance to FPQs." From the snippets I could not tell whether true-premise questions were scored as an over-correction control. [explicit / gap] — [arXiv 2407.05868](https://arxiv.org/abs/2407.05868); [ACL Anthology](https://aclanthology.org/2025.coling-main.698/)

**Pregnant Questions (Srikanth et al., NAACL 2024): medical domain**
- 2,727 inferences (presuppositions and implicatures) from 500 maternal/infant-health questions. GPT-3.5 missed ~63% of expert-identified inferences. Experts preferred inference-augmented answers 75% of the time. False beliefs often appear as implicatures rather than presuppositions. I saw no snippet on over-correction of valid assumptions. [snippet] — [ACL Anthology](https://aclanthology.org/2024.naacl-long.403/); [arXiv 2311.09542](https://arxiv.org/pdf/2311.09542)

**Wang & Blanco, "Identifying and Answering Questions with False Assumptions: An Interpretable Approach" (EMNLP 2025, outside window)**
- Generates and validates atomic assumptions with external evidence. Reported F1 is 0.86 on (QA)², 0.88 on CREPE and 0.87 on FalseQA. Since F1 is reported, both classes are presumably scored; I did not verify any side-effect analysis. [snippet] — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1228/); [arXiv 2508.15139](https://arxiv.org/abs/2508.15139)

### Inferences
- Across 2021–2024 benchmarks, "over-flagging valid questions" is mostly visible as **model-level response bias** (Llama-2's always-false-assumption tendency) inside balanced or paired evaluations. It is not studied as a side effect of a correction intervention, and none of these papers explains it mechanistically.

### Gaps
- Per-class numbers for Syn-QA2, PreWoMe's normal-question scores and KG-FPQ's TPQ handling could not be confirmed. Full text was blocked.

---

## Q6. Other 2021–2024 papers that explicitly discuss over-correction on valid premises

### Takeaway
Two classic-period papers report an intervention-induced trade-off outright:
- **FreshLLMs (Vu et al., 2023/ACL Findings 2024):** a "premise check" instruction helps false-premise questions but "can hurt accuracy on valid premise questions".
- **CoCoNot (Brahman et al., NeurIPS 2024 D&B):** direct fine-tuning for noncompliance, which includes false presuppositions, leads to over-refusal. The fixes are LoRA and preference tuning on a contrast set.

Both observe and mitigate; neither gives a mechanism. The paper that explicitly claims a **cause** is post-window: Wang, Shwartz & Gonen (arXiv 2608.06539, Aug 2026). It attributes the FPQ↔TPQ trade-off to weak fact-checking modules that reject true presuppositions they cannot verify. It also includes the 57%-of-sound-questions-contested finding I could not attribute (see Cited Findings).

### Cited Findings
**FreshLLMs / FreshQA (Vu et al., arXiv Oct 2023; ACL Findings 2024)**
- (a) FreshQA: 600 questions (never-, slow- and fast-changing, plus false-premise). False-premise answers must explicitly point out the false premise. Scored under strict and relaxed criteria. [explicit] — [arXiv 2310.03214](https://arxiv.org/html/2310.03214v1)
- (c) It reports valid-premise accuracy separately. For example, GPT-4 + FreshPrompt gains +30.5% (strict) and +9.9% (relaxed) on valid-premise questions involving pre-2022 knowledge. [snippet] — [arXiv 2310.03214](https://arxiv.org/html/2310.03214v1)
- (d) Ablation: adding "Please check if the question contains a valid premise before answering" improves false-premise accuracy by +23.4% (GPT-3.5) and +6.4% (GPT-4) under strict, and +22.6% / +11.3% under relaxed. But "premise check boosts accuracy on false-premise questions but can hurt accuracy on those with valid premises." **The size of the valid-premise drop was not extractable from snippets.** One summary claims the drop is "significant" for GPT-3.5, which is not verified. [explicit qualitative claim; magnitude unverified] — [ACL Findings PDF](https://aclanthology.org/2024.findings-acl.813.pdf); [arXiv PDF](https://arxiv.org/pdf/2310.03214)
- (e) No mechanism given, per snippets. FreshPrompt's default omits the premise check, which amounts to mitigation by design choice. [implied]

**CoCoNot, "The Art of Saying No: Contextual Noncompliance in Language Models" (Brahman et al., NeurIPS 2024 D&B)**
- (a) A noncompliance taxonomy that includes "false presuppositions" under incomplete requests. It comes with a **contrast set** of benign look-alike queries that should be answered, built for false presuppositions, underspecification, modality limits and safety. [explicit] — [arXiv 2407.12043](https://arxiv.org/pdf/2407.12043); [HF dataset card](https://huggingface.co/datasets/allenai/coconot)
- Size: 1,001 evaluation and 11,477 SFT examples in the original set; 379 contrast evaluation and 927 preference (DPO) examples. [explicit] — [HF dataset card](https://huggingface.co/datasets/allenai/coconot)
- (d): "While direct finetuning of instruction-tuned models can lead to over-refusal and a decline in general capabilities, using parameter efficient methods like low rank adapters helps strike a good balance"; "preference tuning can be effective at reducing overrefusals." For false presuppositions and underspecified requests, models before intervention tend to assume intent and answer directly; compliance on incomplete/unsupported requests is as high as 30% even for GPT-4, Claude, Mixtral and Llama-3-70B. [explicit, snippet] — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/58e79894267cf72c66202228ad9c6057-Paper-Datasets_and_Benchmarks_Track.pdf)
- (e): Observation plus mitigation (LoRA, contrast-set DPO). No mechanism found. Over-refusal numbers by category were not extracted.

**Post-window but directly on point (noted for the 2025+ researcher)**
- **Wang, Shwartz & Gonen, "Don't 'Well, Actually' Me Unless You Know What You're Talking About: Weak Presupposition Verification Degrades General QA Performance" (arXiv 2608.06539, Aug 2026)**
  - Tests the extract → fact-check → respond pipeline across four benchmarks, five model families and several evidence settings.
  - Finding: "methods that perform better on FPQs tend to perform worse on TPQs," and this generalizes across families, sizes, benchmarks and evidence amounts.
  - **Cause stated:** "the result of weak fact checking modules that reject also true presuppositions", i.e., "the fact-checking step rejects true presuppositions it cannot verify."
  - When results are weighted by a realistic ~13% FP rate, plain direct QA is best overall and the most aggressive FPQA method is worst.
  - It also critiques benchmarks for over-representing FPQs.
  - [snippet] — [arXiv HTML](https://arxiv.org/html/2608.06539); [arXiv abs](https://arxiv.org/abs/2608.06539)
- Another snippet (source uncertain: either "Knowing but Not Correcting", [arXiv 2605.05957](https://arxiv.org/html/2605.05957v2), or a related 2025–26 paper) states that when instructed to challenge premises, models "contest 57% of sound (valid) questions… unable to tell which premises are false, they comply by contesting indiscriminately." **Attribution unverified.**
- Others seen but not examined: PCBench "Don't Take the Premise for Granted" ([arXiv 2505.23715](https://arxiv.org/pdf/2505.23715)); "Two Axes of LLM Abstention" ([arXiv 2607.08456](https://arxiv.org/pdf/2607.08456)); MedRedFlag ([arXiv 2601.09853](https://arxiv.org/pdf/2601.09853)).

### Inferences
- **Summary across the classic literature:**

| Paper | Valid-premise questions measured? | Correction-induced harm? | Explanation given |
|---|---|---|---|
| Kim 2021 | Implicitly | Not shown | — |
| (QA)² | Yes | Not framed as such | — |
| CREPE | Yes (macro-F1) | Not shown | Retrieval bottleneck for detection |
| FalseQA | Yes | **Yes** (fine-tuning) | "Catastrophic forgetting"; mitigated by replay |
| FreshLLMs | Yes | **Yes** (prompting) | None; mitigated by omitting the check |
| CoCoNot | Yes (contrast set) | **Yes** (SFT over-refusal) | None; mitigated by LoRA / DPO |
| Syn-QA2 | Yes (paired) | Model bias (Llama-2 over-flags) | Speculative "linguistic structure" |
| PreWoMe | Yes | Claims none | — |
| FAITH | Apparently not | Not reported | Mechanism for FP hallucination only (false-premise heads), not for over-correction |

- None of the 2021–2024 papers I could check gives an internal or causal mechanism for over-correction on valid premises. The only mechanism-style account found, "weak verifiers reject unverifiable true presuppositions", comes from 2026, and its evidence appears to be a behavioral/error analysis of the pipeline, not model internals.
- A plausible synthesis these papers support but do not test: over-correction shows up whenever the system's decision to challenge is not tied to actual knowledge that the premise is false. That decision may be a learned prior to rebut (FalseQA without replay, CoCoNot SFT), an instruction (FreshLLMs premise check), or a verifier that treats "can't verify" as "false" (Wang et al. 2026). This is my inference, not a claim made in any of these papers.

### Gaps
- FreshLLMs' exact valid-premise drop under premise check, CoCoNot's per-category over-refusal numbers, and FalseQA's no-replay rebuttal rate are all unextracted, because full texts were blocked.
- I found no 2021–2024 paper that does internal analysis (probing or attention) of *over-correction* on valid premises. This is a search-limited negative, not proof of absence.
