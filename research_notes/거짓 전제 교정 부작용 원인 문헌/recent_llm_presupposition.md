# Recent (2025–2026) LLM work on false premises: the FPQ-correction vs. valid-premise (NFP/TPQ) trade-off, and whether anyone explains its cause

> **Read this first: how these notes were gathered.** The network egress policy for this session blocked direct fetches of every paper mirror I tried: arxiv.org, export.arxiv.org, alphaxiv, huggingface.co, papers.cool, pith.science, awesomepapers.io, en.papernotes.org, lacuna, openreview and r.jina.ai all returned EGRESS_BLOCKED or 403. GitHub API access to the paper repos was also not enabled. **Every finding below comes from search-engine result summaries and snippets (WebSearch, "extended" mode) of the arXiv abstract, HTML or PDF pages. I read no full texts.** Treat numbers as "reported in snippet" and confirm them against the PDFs before quoting them in a paper. Where a snippet is the only source for a claim, I say so. URLs point to the canonical arXiv/ACL pages that the snippets came from.

---

## Q1. "Don't 'Well, Actually' Me…" (arXiv 2608.06539): what cause is given for rejecting true presuppositions?

### Takeaway
Wang, Shwartz & Gonen do show a consistent trade-off: better FPQ handling comes with worse TPQ (normal-question) handling. They attribute it to the fact-checking module: it rejects presuppositions it **cannot verify**, which amounts to treating "unverifiable" as "false". In threshold terms, a verifier with weak discrimination is run with a reject-by-default criterion. Going by snippets, the explanation is component-level and behavioral: errors are traced to the pipeline stage. I found no evidence of an internal (representation-level) account, or of an account of why verification fails on true presuppositions beyond "weak verification / can't verify".

### Cited Findings
- Full title: "Don't `Well, Actually' Me Unless You Know What You're Talking About: Weak Presupposition Verification Degrades General QA Performance". Authors: Shenran Wang, Vered Shwartz, Hila Gonen. Submitted Aug 6, 2026. — [arXiv abs](https://arxiv.org/abs/2608.06539)
- (a) Domain/datasets: "four benchmarks covering factual knowledge, naturally-occurring information seeking questions, and medical knowledge". CREPE (Reddit ELI5, about 1/4 false presuppositions) is named explicitly. The medical benchmark is not named in the snippets. The brief said it was Cancer-Myth, which is plausible but **unverified**. Also "multiple model families [five], sizes, FPQA methods, and RAG variations". — [arXiv HTML](https://arxiv.org/html/2608.06539)
- (b) Method under test: the "standard prescription": "extract presuppositions, fact-check each one, then respond". — [arXiv HTML](https://arxiv.org/html/2608.06539)
- (c) Valid-premise questions are measured as TPQs ("ordinary true-presupposition questions"). — [arXiv abs](https://arxiv.org/abs/2608.06539)
- (d) Trade-off: "methods that perform better on FPQs tend to perform worse on normal questions". "Improving performance on FPQs consistently hurts performance on TPQs, resulting in overall lower expected QA accuracy when taking into account a realistic FPQ-TPQ distribution." — [arXiv abs](https://arxiv.org/abs/2608.06539)
- (d) Size: per a search summary, the fact-checking step, "while rejecting almost every false presupposition, is also rejecting **more than half the true presuppositions**". Snippet-level only; per-dataset and per-model numbers were not retrievable. — [arXiv HTML (snippet)](https://arxiv.org/html/2608.06539)
- (e) Stated cause: "weak fact checking modules that reject also true presuppositions". Also "the fact-checking step rejects true presuppositions it cannot verify" / "the fact-checking component often overly rejects presuppositions it is unable to verify". — [arXiv abs](https://arxiv.org/abs/2608.06539); [arXiv HTML (snippet)](https://arxiv.org/html/2608.06539)

### Inferences
- The causal claim is one level below the headline result. The degradation is localized to the verification stage, not to presupposition extraction or answer generation, and the failure mode is described as "unverifiable → rejected", a decision-rule/default problem. It is not described as "the verifier believes the true claim is false". This is close to a threshold-vs-discrimination framing, but on the snippets I saw the paper does not use signal-detection terms or internal probes.
- The phrasing "weak … verification" plus testing several "evidence settings / RAG variations" suggests the paper looks at whether better evidence fixes the verifier. Whether RAG closes the gap is not visible in the snippets (gap).
- The paper does not appear to examine whether **decomposition itself** changes the claim, for example an extracted presupposition being stronger or more literal than what the question actually presupposes, so that a "true" question yields a "false" sub-claim. That remains an open alternative mechanism to check in the full text.

### Gaps
- Exact per-benchmark FPQ/TPQ numbers, the identity of the medical benchmark (Cancer-Myth?), the models, and whether an error analysis separates "verifier says false" from "verifier says unverifiable" could not be retrieved, because full-text access was blocked.
- Whether the authors go beyond "can't verify → reject" (e.g., claim-specificity effects of extraction, prior/base-rate effects) is unknown.

---

## Q2. "Two Axes of LLM Abstention: Answer Correctness and Question Answerability" (arXiv 2607.08456): setup, and does it explain why the output doesn't use internal discrimination?

### Takeaway
This single-author preprint (Benedikt J. Wagner, City St George's, University of London) is the clearest **threshold-vs-discrimination** account of the over-challenge side effect I found. A "check the premises" instruction makes the model contest sound premises as well as false ones (57% false challenges), because the output channel cannot discriminate them ("the instruction moves the model's threshold, not its knowledge"). A hidden-state probe *does* discriminate (AUROC 0.69–0.77 on CREPE), and gating the instruction on the probe roughly triples challenge precision. The paper therefore shows that discrimination exists internally but is not expressed. From the snippets, it does **not** give a mechanistic account of *why* the output fails to use that internal signal. That part is a demonstration plus a workaround (routing), not a mechanism.

### Cited Findings
- Author/affiliation: Benedikt J. Wagner, City St George's, University of London. — [arXiv PDF](https://arxiv.org/pdf/2607.08456)
- Framing: selective answering "is usually implemented by thresholding one confidence score", which "conflates two failure modes: emitting a wrong answer to an answerable question, and answering a question that should not be answered at all". — [arXiv abs](https://arxiv.org/abs/2607.08456)
- Models: "five instruction-tuned models from three families (2B to 14B)": Gemma 2 2B, Qwen2.5-3B, Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B. — [arXiv PDF (snippet)](https://arxiv.org/pdf/2607.08456)
- Data: naturally occurring false presuppositions from CREPE, plus the paper's main constructed set(s), which were not named in the snippets. — [arXiv PDF (snippet)](https://arxiv.org/pdf/2607.08456)
- Two axes: "Ordinary answer-confidence tracks whether an answer is right but is nearly blind to whether the question is answerable; a linear probe on hidden states does the reverse." — [arXiv HTML (snippet)](https://arxiv.org/html/2607.08456v1)
- Output vs. internal signal on CREPE: "answer-confidence, P(IK), P(True), and even asking the model outright whether a premise is false all stay near chance, while a hidden-state probe reaches 0.69 to 0.77 AUROC". Another snippet: "trained output readouts sit at or near chance, elicited self-assessments reach at best 0.67, and internal linear readouts 0.69–0.77". — [arXiv HTML (snippet)](https://arxiv.org/html/2607.08456v1); [arXiv PDF (snippet)](https://arxiv.org/pdf/2607.08456)
- (d) Over-challenge: "Instructing a model to check premises backfires, because it then disputes sound and false premises alike (**57% false challenges**), unable to tell them apart." One search summary gives the fuller version, "contesting 74% of false premises but also 57% of sound questions, inventing premise-denials for questions with nothing wrong", with examples of invented denials: "it is not mandatory to have headlights on during the day", "Neosporin is not an antibiotic". **The 74% figure appeared in only one search summary, so treat it as unverified.** — [arXiv abs](https://arxiv.org/abs/2607.08456); [arXiv PDF (snippet)](https://arxiv.org/pdf/2607.08456)
- (e) Explanation offered: "the instruction moves the model's threshold, not its knowledge … unable to tell which premises are false, it complies by contesting indiscriminately". — [arXiv PDF (snippet)](https://arxiv.org/pdf/2607.08456)
- Mitigation: "routing the same instruction with the probe roughly **triples challenge precision**". The router uses an admissibility probe: "challenging only when the admissibility score is low … restores selectivity". Policy: answer only when separate answerability and correctness scores each clear their own threshold; this "certif[ies] both budgets at 0.75 coverage of correct answers, against 0.31 for a single threshold; at 14B it is the only policy that certifies at all". — [arXiv abs](https://arxiv.org/abs/2607.08456); [arXiv PDF (snippet)](https://arxiv.org/pdf/2607.08456)

### Inferences
- Level of explanation: the paper goes one level below "prompting causes over-challenge". It localizes the problem as **output-channel non-discrimination plus a criterion shift**, with internal evidence: the probe AUROC is above the output readouts. That fits "threshold vs discrimination with internal evidence". It stops at showing the dissociation. The snippets contain no circuit, attention or patching analysis of *why* the generation pathway ignores the probe-decodable signal.
- The probe AUROC of 0.69–0.77 is modest. "Restores selectivity" therefore probably means "improves precision substantially", not "eliminates false challenges". The absolute post-routing false-challenge rate was not visible.

### Gaps
- The exact challenge-prompt wording, per-model 74%/57% breakdown, post-routing false-challenge rate and in-distribution datasets were not retrievable.
- I found no evidence that the paper tests *why* the internal signal is not used (e.g., layer-wise readout, causal patching).

---

## Q3. Cheng, Hawkins & Jurafsky, "Accommodation and Epistemic Vigilance" (arXiv 2601.04435, ACL 2026): are false positives on valid content measured, and does "wait a minute" increase them?

### Takeaway
Yes, false positives are measured. The paper uses the 150-question Cancer-Myth-NFP set, where no correction should occur, and reports corrected scores (main metric minus FPR). The authors claim "wait a minute" and similar pragmatic interventions "significantly improve performance … **while preserving low false-positive rates**." Their causal story is a **pragmatic/surface-cue** account of accommodation: at-issueness, linguistic encoding and source reliability shift how much the model accommodates. That explains *why models fail to challenge*, and it predicts that cue manipulations shift challenge behavior. I found no internal or mechanistic analysis, and no account of the over-challenge side of the trade-off beyond the FPR controls.

### Cited Findings
- Authors/venue: Myra Cheng, Robert D. Hawkins, Dan Jurafsky; ACL 2026 (long, reportedly oral). — [ACL Anthology](https://aclanthology.org/2026.acl-long.736/); [arXiv](https://arxiv.org/abs/2601.04435)
- (a) Benchmarks: Cancer-Myth and SAGE-Eval (misinformation), ELEPHANT (social sycophancy). — [arXiv PDF](https://arxiv.org/pdf/2601.04435)
- (e) Account: failures are "consequences of LLMs defaulting to accommodating users' assumptions and exhibiting insufficient epistemic vigilance". The factors that drive human accommodation (at-issueness, linguistic encoding, source reliability) "similarly affect accommodation in LLMs, explaining performance differences across three safety benchmarks". — [arXiv abs](https://arxiv.org/abs/2601.04435)
- (b) Intervention: "simple pragmatic interventions, such as adding the phrase 'wait a minute', significantly improve performance on these benchmarks while preserving low false-positive rates." — [arXiv abs](https://arxiv.org/abs/2601.04435)
- (c) FP control: "For an intervention to be useful, it should avoid challenging the user when unnecessary". They use "a 150-question non-false-presupposition (NFP) dataset where no correction should occur, calculating a corrected score by subtracting the false positive rate (FPR) from the main metric". — [arXiv PDF (snippet)](https://arxiv.org/pdf/2601.04435)
- Code: [GitHub myracheng/accommodation](https://github.com/myracheng/accommodation) (I could not read it; API access was not enabled).

### Inferences
- (d): the authors report that FPR stays *low*. Whether "wait a minute" *raises* FPR at all (e.g., from near 0 to a few percent) could not be seen, because no per-condition FPR numbers appeared in the snippets. "Preserving low FPR" is consistent with a small increase.
- Compare Cancer-Myth's own mitigation (precautionary prompt + GEPA), which produced 41% NFP false positives (Q5). That makes Cheng et al.'s claim notable: a cue-level intervention ("wait a minute") moved correction without the large over-challenge that a generic "check for false premises" instruction caused. One possible reading is that the pragmatic cue changes *how the presupposition is processed* (less backgrounded, more at-issue) rather than simply lowering a global challenge threshold. **This is my inference; the paper does not test it internally.**
- The NFP set is only 150 questions and is used to estimate FPR across interventions, so small FPR differences will have wide CIs.

### Gaps
- Exact FPR per model and intervention, and the size of the Cancer-Myth gain, were not retrievable.
- No internal or representation analysis was found.

---

## Q4. "Knowing but Not Correcting: Routine Task Requests Suppress Factual Correction in LLMs" (arXiv 2605.05957): do CDS and DPA cause over-correction on true statements? Is there a mechanism?

### Takeaway
The paper has a genuine mechanistic analysis, but of the **suppression** side, not of over-correction. The model registers the error internally regardless of output; task context diverts early-layer attention from the false claim; and output intent "crystallizes toward compliance at middle layers". Suppression therefore happens at response selection, not knowledge encoding. Its interventions are CDS (correction-direction steering) and DPA. From the snippets, I could find **no measurement of over-correction on true statements** for CDS or DPA. The capability side-effect it reports is reasoning preservation: DPA "is the only method that preserves or improves reasoning capability on both models". Whether a true-statement control exists is unverified.

### Cited Findings
- Authors: Z. Chen, H. Lin, Z. Chen, Y. Tian, G. Yang, D. Wang, Y. Guo, H. Zhu, J. Cheng. — [arXiv PDF](https://arxiv.org/pdf/2605.05957)
- Phenomenon: LLMs correct false claims in isolation but comply when the same claims are embedded in task requests. Suppression rates are 19%–90%, with four models above 80%. The trigger is benign task framing (role statements, output specifications, compliance cues). — [arXiv HTML v2](https://arxiv.org/html/2605.05957v2)
- (e) Mechanism (for suppression): hidden-state, uncertainty and attention analysis shows "the model registers the error internally regardless of output behavior, but task context diverts early-layer attention from the false claim as output intent crystallizes toward compliance at middle layers … suppression occurs at response selection rather than at knowledge encoding." — [arXiv HTML v2](https://arxiv.org/html/2605.05957v2)
- (b) Methods/results: CDS raises correction on Qwen3.5-9B from 0% to 58.2% (78/134); DPA reaches 32.8% (44/134). A random-direction control achieves about 10% vs 50% for the learned direction. CDS does better on fabricated events and future claims, where internal knowledge strongly contradicts the claim; DPA does better on dates and numbers embedded in plausible statements. — [arXiv PDF (snippet)](https://arxiv.org/pdf/2605.05957); [arXiv HTML v2](https://arxiv.org/html/2605.05957v2)
- (d) Side effects: "DPA … is the only method that preserves or improves reasoning capability on both models", which implies CDS and other baselines cost some reasoning capability. — [arXiv HTML v2](https://arxiv.org/html/2605.05957v2)

### Inferences
- The finding that CDS, a single static direction, "shifts the model toward fact-checking mode" is the kind of global intervention that would be expected to raise challenges on true claims as well (cf. Dual-Stance, Q6). Without a true-statement control, though, I cannot say whether it does. This is a clear evaluation gap relative to Well Actually, Two Axes and Cancer-Myth.
- The mechanism here ("knows internally; response selection overrides") is the mirror image of Two Axes ("internal discrimination exists; output criterion shifts indiscriminately"). Together they suggest that both under- and over-correction are failures of the **judgment→response mapping**, not of knowledge. That synthesis is mine, not either paper's.

### Gaps
- No data on CDS/DPA false-correction rates on true or valid statements was found in the snippets. Check the full paper for a "true-claim" or "control" condition.

---

## Q5. Other benchmarks and evaluations: Cancer-Myth, Sathyanathan et al., PCBench, RPCBench, MMPCBench, FPCO-Dialog, Prior Audit-Repair, medical misleading-context work

### Takeaway
Among the benchmark papers, **Cancer-Myth (ICLR 2026 version)** gives the most direct quantitative evidence of the trade-off on valid medical questions: precautionary prompting + GEPA raises Cancer-Myth to about 80% but misflags 41% of Cancer-Myth-NFP and causes about a 10% relative drop on other medical benchmarks. It offers no causal explanation beyond the observation (a "whack-a-mole" framing in a review). **FPCO-Dialog** builds an over-correction metric (CorrFP@K on premise-correct turns) into a VLM benchmark. **Prior Audit-Repair** is not about premises, but it is the cleanest signal-detection demonstration that a context manipulation moves the **criterion, not d′**. PCBench, RPCBench and MMPCBench mostly measure detection (under-critique). MMPCBench reports a "consistency gap": errors are identified in reasoning but suppressed in the output. None of these explains *why* stronger checking damages valid-premise questions.

### Cited Findings

**Cancer-Myth (Zhu et al.; arXiv 2504.11373; ICLR 2026)**
- (a) 585 expert-verified cancer questions with false presuppositions. No frontier LLM (GPT-5, Gemini-2.5-Pro, Claude-4-Sonnet) corrects more than 43%. "Even advanced medical agentic methods do not prevent LLMs from ignoring false presuppositions." — [arXiv HTML v2](https://arxiv.org/html/2504.11373v2); [OpenReview](https://openreview.net/forum?id=fOXLhZIaUj)
- (c) Cancer-Myth-NFP: 150 questions that physicians confirm have no false presupposition, used to measure false positives. — [OpenReview forum](https://openreview.net/forum?id=fOXLhZIaUj); [Liner review](https://liner.com/review/cancermyth-evaluating-large-language-models-on-patient-questions-with-false)
- (d) "Typical mitigation strategies, such as adding precautionary prompts with GEPA optimization, can raise accuracy on Cancer-Myth to 80%, but at the cost of misidentifying presuppositions in **41% of Cancer-Myth-NFP** questions and causing a **10% relative performance drop on other medical benchmarks**." — [OpenReview forum](https://openreview.net/forum?id=fOXLhZIaUj) (from a search summary of the ICLR version; confirm in the PDF)
- (e) The explanation offered is behavioral: models "possess the correct medical facts but fail to apply them", prioritizing a "polite assistant" role. That accounts for under-correction. No cause is given for the NFP false-positive side. — [Moonlight review](https://www.themoonlight.io/en/review/cancer-myth-evaluating-ai-chatbot-on-patient-questions-with-false-presuppositions) (secondary source)

**Sathyanathan, Vasisht & Pruthi, "Evaluating Reasoning Models for Queries with Presuppositions" (arXiv 2605.03050; Findings ACL 2026)**
- (a) Queries with "varying degrees of presuppositions" across health, science and general knowledge. Reasoning models gain 2–11% in accuracy but still fail to challenge 26–42% of false presuppositions. — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1201/); [arXiv](https://arxiv.org/abs/2605.03050)
- (e) Qualitative cause for *under*-correction: in LRMs, "early factual inaccuracies introduced during reasoning cascade through subsequent steps". They also observe deceptive-looking selective evidence presentation. — [arXiv](https://arxiv.org/abs/2605.03050)
- (c)/(d): one snippet says "the neutral region is smaller when reasoning is enabled", which suggests a graded presupposition-strength design with a neutral band. I found no explicit over-challenge numbers on true or neutral queries (snippet only; unverified). — [arXiv HTML (snippet)](https://arxiv.org/html/2605.03050)

**PCBench, "Don't Take the Premise for Granted" (arXiv 2505.23715; Findings EMNLP 2025)**
- (a) 1,200 base problems (4 error types × 3 difficulty levels × 100), expanded to 3,600 with original, flawed, and flawed-with-instruction variants; 15 LLMs. — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.44/)
- Findings: models "rely heavily on explicit prompts to detect errors"; reasoning ability does not consistently correlate with premise critique; flawed premises trigger overthinking. (c): the "original" (unflawed) variant exists. I found no report of whether explicit critique instructions cause false critiques on originals (gap). — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.44/)

**RPCBench (arXiv 2609.00918)**
- 4,623 instances, 5 recommendation domains, 10 premise-failure types, 11 LLMs. Detection is the bottleneck (51.5% average). Critique quality peaks at intermediate reasoning length (an "overthinking penalty"). Over-challenge on valid requests was not reported in the snippets. — [arXiv HTML](https://arxiv.org/html/2609.00918)

**MMPCBench (arXiv 2608.29286)**
- 4 primary error types / 12 subcategories; 14 MLLMs. "Consistency gap": "reasoning models can often correctly identify and analyze errors during internal reasoning yet suppress these valid insights in final outputs to prioritize response compliance." This is a reasoning-trace-level (not hidden-state) judgment-vs-response dissociation, on the under-correction side. — [arXiv HTML](https://arxiv.org/html/2608.29286); [OpenReview PDF](https://openreview.net/pdf/f2243ab6b0e1438f353224dd61aa87d62064d276.pdf)

**FPCO-Dialog (arXiv 2609.03331; EMNLP 2026)**
- (a) 1,080 MS COCO / Open Images images, 10,800 turns. Ten-turn dialogues: turns 1–3 premise-correct, turns 4–10 repeat one false referring expression (identity, attribute or location errors). 20 VLMs. — [arXiv HTML](https://arxiv.org/html/2609.03331)
- (c) Over-correction metric: **CorrFP@K**, the correction rate on premise-correct turns "for checking indiscriminate over-correction". Cooperation without correction "is not itself an error label". — [arXiv HTML](https://arxiv.org/html/2609.03331)
- Results: large cross-model differences in correction tendency; identity errors are corrected most and location errors accommodated most. CorrFP@K values and any correction/over-correction correlation were not retrievable. — [arXiv HTML](https://arxiv.org/html/2609.03331)

**"Prior Audit-Repair Context Shifts LLM Verifier Thresholds Toward Leniency" (arXiv 2608.16003; Mazaheri & Mazaheri)**
- A completed audit-repair episode in context lowers false alarms in 15/15 model × wording combinations, by 2.8–11.5 pp versus a length-matched control. "Signal-detection analysis located the change in the threshold rather than in discrimination — the criterion moves in 15 of 15 combinations and survives correction in 13 while d′ survives in none." — [arXiv abs](https://arxiv.org/abs/2608.16003)
- Relevance: it is not about premises, but it is a methodological template for decomposing over-challenge into criterion vs d′ (behavioral SDT, no internals). Its warning ("the threshold moved without anyone asking it to") generalizes to premise-checking prompts. — [arXiv abs](https://arxiv.org/abs/2608.16003)

**Medical misleading context: Linzmayer & Elhadad, "Untangling the Mechanisms of Misleading Context in Medical QA" (arXiv 2609.02754; ML4H 2026)**
- MedMisBench reasoning subset (8,627 questions), with injected fabricated evidence vs bare assertion. Three reasoning models are 10–27 points more likely to adopt a bare assertion. Trace resampling shows evidence "entering early and accumulating while the assertion redirects the conclusion near its end". A trace monitor catches 78% of corrupted decisions at 5% FPR, versus at most 32% from responses. This concerns the acceptance side, not over-challenge. — [arXiv abs](https://arxiv.org/abs/2609.02754)

**DEDUCE / Mis-FactQA, "From Passive Response to Proactive Correction" (arXiv 2608.25894)**
- A detect (fine-grained fact extraction + verification) → deliberate → correct framework, with a new Mis-FactQA dataset. It "significantly improves both accuracy and error correction". Whether clean inputs were tested for over-correction was not visible. — [arXiv HTML](https://arxiv.org/html/2608.25894)

### Inferences
- Cancer-Myth's 41% NFP false positive under GEPA-optimized precautionary prompting, Wagner's 57% false challenges under a premise-check instruction, and Well Actually's ">half of true presuppositions rejected" are three independent reports of the same phenomenon, all of a similar order. Generic "be vigilant" instructions shift the challenge criterion far more than they improve discrimination.
- Optimizing a prompt only on the FPQ set (GEPA on Cancer-Myth) is expected to find a lenient-to-challenge criterion; the side effect is a predictable consequence of the objective. This is my inference, not stated in the paper.

### Gaps
- No full-text numbers for FPCO-Dialog CorrFP@K, PCBench over-critique on originals, or Sathyanathan's neutral/true-presupposition behavior.
- The Cancer-Myth 41%/10% figures come from a search summary of the ICLR/OpenReview version; they are not in the v1 arXiv abstract I saw.

---

## Q6. Does any 2025–2026 paper identify the mechanism by which stronger premise checking damages valid-premise questions?

### Takeaway
I found no paper that gives a full internal mechanism (circuit or causal) for *why stronger premise-checking hurts valid-premise questions*. The closest are below, ordered by depth of explanation.
1. **Two Axes (2607.08456)**: threshold-not-knowledge, with internal evidence (probe discriminates, output doesn't). It does not show why the output fails to use the internal signal.
2. **Dual-Stance Evaluation of Sycophancy (2606.11205)**: a real geometric mechanism in the analogous sycophancy setting. Sycophantic and factual agreement live in distinct subspaces, but the steering direction projects equally onto both, so de-sycophancy steering also cuts agreement with correct statements by 14%.
3. **Well Actually (2608.06539)**: component-level cause. The verifier rejects what it cannot verify.
4. **Verifier-strictness steering (2605.20745)** and **Prior Audit-Repair (2608.16003)**: strictness/criterion is a separable latent or behavioral variable, and uniform shifts trade error detection against false rejection.
5. **Activation-bottleneck (2608.12321)**, **Decodable-but-not-corrected (2605.05715)** and **Knowing-but-not-correcting (2605.05957)**: "knowledge is encoded but routing/response selection doesn't use it", with representational entanglement limiting fixed steering. These are adjacent mechanistic accounts, but none studies over-challenge of valid premises.

### Cited Findings
- **Dual-Stance Evaluation of Sycophancy (arXiv 2606.11205)**: Llama-3-8B-Instruct, centroid-difference steering. "The model represents sycophantic and factual agreement in geometrically distinct subspaces, yet the steering direction projects equally onto both and cannot differentially target either." Steering "reduces sycophantic agreement by 89% but also reduces agreement with factually correct statements by 14%". The method tests both stances ("the Earth is flat" / "the Earth is round"). — [arXiv HTML](https://arxiv.org/html/2606.11205v1)
- **Two Axes (2607.08456)**: output readouts are near chance on CREPE, internal probes reach 0.69–0.77 AUROC; the instruction "moves the model's threshold, not its knowledge"; probe routing triples challenge precision. — [arXiv PDF](https://arxiv.org/pdf/2607.08456)
- **The Hidden Signal of Verifier Strictness / VerifySteer (arXiv 2605.20745)**: generative verifiers "may be under-critical and miss erroneous steps, or over-critical and reject correct reasoning". "Verifier strictness is encoded in a verification-specific latent signal" at the paragraph-boundary delimiter token. "Uniform steering induces a trade-off between error detection and correctness certification", which is fixed by sample-level routing with latent correctness signals (ProcessBench, Hard2Verify). — [arXiv HTML](https://arxiv.org/html/2605.20745v1)
- **Playing Devil's Advocate (arXiv 2605.21006)**: per a search summary, behavior-specific sycophancy vectors (CAA) may "over-correct on simple factual claims", scoring below baseline on factual accuracy, while critical-role persona steering gives "calibrated disagreement". Snippet only. — [arXiv PDF](https://arxiv.org/pdf/2605.21006)
- **LLMs Know the Constraint But Do Not Use It (arXiv 2608.12321; Li, Krishnan, Padman, CMU)**: when a "salient surface cue competes with an implicit feasibility constraint", the constraint is encoded (probes >88%) symmetrically across constraint-present and -absent prompts but "only sometimes routed into the decision". Activation patching repairs one model (+6.4 nats) and not the other (−0.07). "Aggregate accuracy conflates genuine constraint inference with conservative defaulting." — [arXiv HTML](https://arxiv.org/html/2608.12321)
- **Decodable but Not Corrected (arXiv 2605.05715; Ming Liu)**: in medical QA "Overthinking", the failure is decodable at 71.6% balanced accuracy, but 29 fixed linear-steering configurations yield no correction. The direction overlaps 85–88% with task-critical computation, and non-targeted steering costs 12.1 pp accuracy ("representational entanglement"). The probe still supports selective abstention (AUROC 0.610). — [arXiv abs](https://arxiv.org/abs/2605.05715)
- **Knowing but Not Correcting (2605.05957)**: suppression arises at response selection (attention diversion in early layers, compliance intent at middle layers), not in knowledge encoding. — [arXiv HTML v2](https://arxiv.org/html/2605.05957v2)

### Inferences
Candidate mechanisms in the literature, mapped to the brief's categories:
- **Threshold vs discrimination with internal evidence**: Two Axes is the clearest case (prompting moves the criterion while discrimination lives only in hidden states). Prior Audit-Repair is a behavioral SDT analogue, and VerifySteer gives a latent-strictness analogue. This is the best-supported explanation of why generic premise-check prompts produce 41–57% false challenges.
- **Shared internal components / entanglement**: Dual-Stance (one steering direction covers both sycophantic and factual agreement) and Decodable-but-not-corrected (85–88% overlap with task computation) are direct evidence that interventions on a shared direction hit valid cases too. Neither is premise-specific.
- **Judgment-vs-response separation**: Knowing-but-not-correcting (internal error registration overridden at response selection), MMPCBench (trace identifies, output suppresses) and Activation Bottlenecks (encoded but not routed) are all on the under-correction side. The over-challenge side is mirror-symmetric in Two Axes.
- **Decomposition changing the claim**: Well Actually's pipeline is the natural place for this (extracted sub-claims judged as "unverifiable"). The snippets attribute failure to verification strength and an unverifiable→reject default, not to extraction changing the claim. **No paper I found tests the decomposition hypothesis directly.**
- **Surface-cue reliance**: Cheng et al. (pragmatic cues such as at-issueness and encoding modulate accommodation) and Activation Bottlenecks (salient surface cue vs constraint) support it. Neither connects cue reliance to false challenges on valid questions.
- **Bottom line for the report writer**: the field has (i) repeated, quantified observations of the trade-off (Cancer-Myth 41% NFP; Two Axes 57%; Well Actually >50% of TPQ presuppositions rejected), (ii) one premise-specific threshold-vs-discrimination account with probe evidence (Two Axes), (iii) one component-level account (Well Actually), and (iv) adjacent mechanistic results (Dual-Stance, VerifySteer, Knowing-but-not-correcting) that suggest, without demonstrating, a shared-direction / criterion-shift mechanism for premise checking. A paper that combines internal probes of premise falsity with causal interventions on Cancer-Myth / Cancer-Myth-NFP to show *why* the challenge decision does not read out the internal discrimination appears to be an open gap as of these searches.

### Gaps
- Not found: any 2025–2026 paper that causally (patching/ablation) explains over-challenge of *valid premises* specifically.
- AbstentionBench (2506.09038) and "LLM Abstention Can Be a Prompt Artifact" (2507.16199) surfaced in searches but were not examined. They may contain relevant over-abstention or prompt-artifact findings.
- All numbers above are from search-result summaries. Full-text verification was impossible in this session because of the egress policy, so cross-check before citing.
