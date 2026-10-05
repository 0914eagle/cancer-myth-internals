# Abstention / unanswerable-question / noncompliance training and its over-abstention side effect on valid questions: what the literature says about the cause (2023–2026)

> **Source-access caveat (applies to the whole file).** In this session every full-text route was blocked by the egress proxy: arxiv.org, ar5iv, export.arxiv, alphaxiv, huggingface.co, openreview, semanticscholar, proceedings.neurips.cc, ojs.aaai.org, liner.com, themoonlight.io, emergentmind and the SDSC paper mirror all returned EGRESS_BLOCKED, and `gh api` was refused for the allenai/noncompliance, facebookresearch/AbstentionBench and shizhediao/R-Tuning repos. **Every finding below therefore comes from search-engine result summaries and snippets (abstract-level), not from reading the papers.** Table-level numbers (contrast-set compliance deltas, over-conservativeness scores, CRaFT over-refusal rates) could not be checked and are listed as Gaps rather than guessed. Anyone who can open the PDFs should fill those numbers in first.

## Q1. R-Tuning (Zhang et al., NAACL 2024): does refusal-aware tuning over-refuse known questions?

### Takeaway
R-Tuning splits training data into questions the model already gets right ("certain") and questions it gets wrong ("uncertain"). It reports better refusal on unknown questions while keeping performance on known ones. Later work reports that R-Tuning *magnifies* over-refusal. CRaFT (AAAI 2025) gives the clearest proposed **cause** of over-refusal in this whole line of work: conflicting supervision on samples that sit close together in representation space, plus a stale knowledge snapshot.

### Cited Findings
- **R-Tuning, (a) domain and (b) method.** Knowledge QA. Before fine-tuning, the pre-trained model is run on the training set. Items it answers correctly become "Certain Data". Items it gets wrong become "Uncertain Data" and are given uncertainty cues such as "I am unsure". The model is then tuned to refuse questions outside its parametric knowledge. — [ACL Anthology entry](https://aclanthology.org/2024.naacl-long.394/); [HF papers page](https://huggingface.co/papers/2311.09677) (snippet only)
- **R-Tuning, (c) and (d).** The paper says R-Tuning "effectively improves a model's ability to answer known questions and refrain from answering unknown questions". I could not see whether a separate over-refusal rate on known questions is reported, or how large it is. — [emergentmind RAIT topic page summarizing R-Tuning](https://www.emergentmind.com/topics/refusal-aware-instruction-tuning) (aggregator)
- **Independent evaluation.** Zhou et al., "Do Retrieval Augmented Language Models Know When They Don't Know?" (AAAI 2026, arXiv 2509.01476) find that LLMs "exhibit significant over-refusal behavior". Of two refusal post-training methods, "the over-refusal problem is mitigated by in-context fine-tuning but **magnified by R-tuning**". They also find that better refusal behavior does not necessarily mean better calibration or higher overall accuracy. — [arXiv 2509.01476](https://arxiv.org/pdf/2509.01476); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40822)
- **CRaFT, Zhu et al., "Utilize the Flow before Stepping into the Same River Twice" (AAAI 2025, arXiv 2410.06913). (e) Proposed cause of over-refusal in refusal-aware instruction tuning (RAIT):**
  - **Static conflict.** Similar samples in the LLM's feature space receive different supervision: some keep their original answer and some are relabeled "I don't know".
  - **Dynamic conflict.** The model's knowledge changes during SFT. Questions that were unanswerable at labeling time become answerable, but they keep their stale "I don't know" label.
  - **Mitigation.** CRaFT filters and modifies data using response certainty to reduce static conflict, and runs a "rehearsal training" pass to track how the knowledge state drifts, which targets dynamic conflict.
  - **Training data.** 5,000 MMLU samples (multiple choice) and 10,000 TriviaQA samples (open-ended).
  - [ResearchGate figure: "Two causes of over-refusal"](https://www.researchgate.net/figure/Two-causes-of-over-refusal-a-Static-conflict-means-the-similar-samples-in-the-LLMs_fig1_384770741); [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/34812/36967); [arXiv 2410.06913](https://arxiv.org/pdf/2410.06913)
- **GRAIT (NAACL 2025, arXiv 2502.05911).** Frames RAIT as two goals: reject unknown questions, and "avoid over-refusal to ensure questions that can be correctly answered are not rejected". It uses gradient-based sample selection plus adaptive loss weighting during fine-tuning to reduce over-refusal. It is a gradient-perspective argument about which training samples cause over-refusal. — [arXiv 2502.05911](https://arxiv.org/html/2502.05911v1)

### Inferences
- CRaFT's "static conflict" is a representation-level mechanism claim. If near-identical representations are pushed toward both "answer" and "I don't know", the decision boundary blurs and refusal spills onto answerable neighbors. This parallels a "surface cue vs. true answerability" account. From the snippets I cannot tell whether the evidence is a direct representation-similarity analysis or mainly the ablation gain from filtering conflicting samples. Treat it as a hypothesis backed by intervention results, not a fully established mechanism.

### Gaps
- I could not get R-Tuning's own numbers for refusal or accuracy on the "certain" (known) set, or Zhou et al.'s magnitude for "magnified" over-refusal (full text blocked).
- I could not get CRaFT's over-refusal-rate (ORR) numbers for R-Tuning vs. vanilla vs. CRaFT, or details of the representation-space evidence for static conflict.

## Q2. CoCoNot / "The Art of Saying No" (Brahman et al., NeurIPS 2024 D&B): contrast-set over-refusal and its explanation

### Takeaway
CoCoNot measures over-refusal directly through a contrast set of benign queries that look like refusable ones. Both system prompts and full fine-tuning on noncompliance data lower compliance on contrast items. The paper's fixes are LoRA and preference tuning (DPO) on contrast data. In the abstract-level material I could access, it **observes and mitigates** over-refusal but does **not** give a mechanistic cause beyond "overfitting to refusing benign requests".

### Cited Findings
- **(a) Domain and dataset.**
  - Taxonomy of contextual noncompliance: incomplete, unsupported, indeterminate, humanizing and unsafe requests. False presuppositions, underspecified requests and unknowns sit under the incomplete, unsupported and indeterminate types.
  - Evaluation set: 1,000 human-verified noncompliance prompts plus a contrastive counterpart.
  - [arXiv 2407.12043 (search snippet)](https://arxiv.org/pdf/2407.12043)
- **Dataset sizes.** 1,001 evaluation and 11,477 SFT training examples in the original set. 379 evaluation and 927 preference examples in the contrast set "for testing and mitigating exaggerated noncompliance". — [HF dataset card allenai/coconot (via search snippet)](https://huggingface.co/datasets/allenai/coconot)
- **(c) Answerable side measured.** Yes: the CoCoNot-Contrast set and the XSTest benign subset (XST_B). — [search summary of paper](https://arxiv.org/html/2407.12043v2)
- **(d) Side effect, direction only.**
  - "On contrast sets (XST_B and CoCoNot-Contrast), there can be a decline in compliance suggesting the model **overfits to refusing benign requests**."
  - System prompts "can lead to over-refusal on benign queries (contrast sets), suggesting that a superficial instruction-based approach is insufficient".
  - Direct (full) fine-tuning "can lead to over-refusal and general performance decline".
  - LoRA "performs much better on contrastive test sets" while improving noncompliance less dramatically.
  - Larger Tulu-2 models show lower overall compliance. Tulu-2-DPO does better at noncompliance than the SFT-only version.
  - [arXiv 2407.12043](https://arxiv.org/pdf/2407.12043); [ResearchGate figure: compliance rate vs. LoRA training data size](https://www.researchgate.net/figure/Compliance-Rate-when-LoRa-finetuning-Tulu-2-7B-on-different-training-data-sizes_fig2_382331618)
- **(e) Explanation.** The only explanation I found is the label "overfitting to refusing benign requests", plus the observation that parameter-efficient training (LoRA) and preference data on contrast items reduce it. That is mitigation, not mechanism. I found no ablation, probing or attention analysis of *why* noncompliance training spreads to contrast items. — same sources as above (abstract-level)

### Inferences
- CoCoNot's design is close to the "contrast set" approach. It shows that over-refusal can be measured and partly reduced by data choice (contrast-set DPO) and by limiting update capacity (LoRA). The LoRA result loosely suggests full fine-tuning overwrites a broad "refuse" disposition rather than learning a narrow answerability rule, but the paper does not test this.

### Gaps
- I could not get the exact contrast-set compliance numbers before and after SFT, LoRA or DPO, or the per-category results for false presupposition and underspecified items. These would come from the paper's tables (full text blocked).

## Q3. Surveys and benchmarks: Know Your Limits, AbstentionBench, prompt-artifact abstention

### Takeaway
- AbstentionBench mainly documents **under**-abstention: reasoning fine-tuning lowers abstention recall by 24%. Its key diagnostic is a **gap between reasoning traces and final answers**.
- The "Abstention Inflation" paper (2507.16199) is the closest to a **causal story for over-abstention on answerable questions**. It argues that models imitate the *surface pattern* of abstention: an uncertainty option, or even a random word in its place, triggers abstention on solvable problems.
- The Wen et al. survey frames the problem but, from snippets alone, I could not confirm that it offers a cause.

### Cited Findings
- **Know Your Limits (Wen et al., TACL 2025, arXiv 2407.18418).** Organizes the abstention literature from three angles (query, model, human values) and covers methods, benchmarks and metrics. I could not see whether it analyzes the cause of over-abstention. — [arXiv 2407.18418](https://arxiv.org/pdf/2407.18418); [TACL](https://transacl.org/index.php/tacl/article/view/7077)
- **AbstentionBench (Kirichenko et al., NeurIPS 2025 D&B, arXiv 2506.09038), (a) and (c).**
  - 20 datasets in the abstract; one version summary says 17. Six scenarios: Answer Unknown, False Premise, Stale Data, Subjective, Underspecified Context, Underspecified Intent.
  - [arXiv 2506.09038](https://arxiv.org/abs/2506.09038); [arXiv HTML v1](https://arxiv.org/html/2506.09038v1)
- **AbstentionBench, (d).** "Reasoning fine-tuning degrades abstention by 24% on average, even for math and science domains." DeepSeek R1 Distill and s1.1 are compared with their instruction-tuned bases.
  - Scaling helps little.
  - A crafted system prompt boosts abstention "but does not resolve models' fundamental inability to reason about uncertainty".
  - False premise and underspecified cases remain hard.
  - [arXiv 2506.09038](https://arxiv.org/pdf/2506.09038)
- **AbstentionBench, (e) partial mechanism.** "Reasoning traces do contain increased expressions of uncertainty, but despite this, models continue to provide a definitive final response." This is error analysis showing a split between internal recognition and output. — [arXiv HTML v1](https://arxiv.org/html/2506.09038v1)
- **"LLM Abstention Can Be a Prompt Artifact, in Addition to Genuine Uncertainty" (Ling et al., arXiv 2507.16199). (d) and (e).**
  - When prompts contain uncertainty elements, LLMs "are inclined to abstain even on problems they are capable of solving". The authors call this **"Abstention Inflation"**.
  - Adding "Unknown" as an extra option causes "serious accuracy drops on True/False Questions". Replacing "Unknown" with an **unrelated random word produces an identical effect**.
  - The authors argue that LLMs "are trained to imitate the surface pattern of abstention, rather than to express genuine uncertainty".
  - This is a controlled-perturbation causal test of a surface-cue account.
  - [arXiv 2507.16199](https://arxiv.org/html/2507.16199v9)
- **Alignment for Honesty (Yang et al., NeurIPS 2024, arXiv 2312.07000).**
  - Defines a **prudence score** (refusing unknowns) and an **over-conservativeness score** (refusing questions the model could answer), combined into an honesty score.
  - SFT variants: ABSOLUTE, CONFIDENCE and MULTISAMPLE.
  - The authors report a low "tax" on helpfulness.
  - This is a metric framework that measures the valid side explicitly. I could not retrieve the per-method numbers or any causal analysis.
  - [arXiv 2312.07000](https://arxiv.org/html/2312.07000v2)

### Inferences
- The prompt-artifact result is the strongest evidence I found for a "surface cue" cause of over-abstention. If an *unrelated token* in the abstention slot inflates abstention just as much, the abstain response is cued by format and not driven by an answerability judgment. That fits a "response is decoupled from judgment" account.

### Gaps
- AbstentionBench is mostly framed around recall on should-abstain items. I could not confirm the size of any false-abstention side effect on answerable controls, for example after system-prompt boosting.
- I could not get the over-conservativeness numbers for Alignment for Honesty.

## Q4. Mechanism-level evidence: internal representations separate answerability or knowledge from the abstain/answer response

### Takeaway
Several papers show that internal states encode answerability or "knownness" better than outputs reveal, and that a separate, steerable direction or routing step produces the refusal response. The strongest causal (intervention-based) evidence is Ferrando et al. (ICLR 2025): steering along SAE entity-recognition directions makes the model **refuse questions about known entities**, a mechanistic route to over-abstention. Reasoning-model papers from 2025–26 explicitly name a "detection-to-abstention gap" or "cognition vs. response misalignment". These are close analogues of "judgment vs. response separation".

### Cited Findings
- **Slobodkin et al., "The Curious Case of Hallucinatory (Un)answerability" (EMNLP 2023, arXiv 2310.11877).**
  - Even when models produce hallucinated answers to unanswerable questions, their hidden states "encode the answerability of an input query". The representation of the first decoded token is often a strong indicator.
  - Answerable and unanswerable questions are "linearly separable in the embedding space".
  - (e) Probing evidence that **internal judgment exceeds the output**.
  - [arXiv 2310.11877](https://arxiv.org/pdf/2310.11877); [ACL Anthology](https://aclanthology.org/2023.emnlp-main.220)
- **Ferrando, Obeso, Rajamanoharan, Nanda, "Do I Know This Entity?" (ICLR 2025, arXiv 2411.14257).**
  - Sparse autoencoder (SAE) latents in middle layers separate known from unknown entities.
  - The directions are "causally relevant: capable of steering the model to **refuse to answer questions about known entities**, or to hallucinate attributes of unknown entities when it would otherwise refuse".
  - The SAEs were trained on the base model, yet the directions control chat-model refusal. The authors read this as "chat finetuning has repurposed this existing mechanism".
  - The directions disrupt attention heads that move entity attributes to the final token.
  - (e) Causal intervention plus attention-level mechanism.
  - [arXiv 2411.14257](https://arxiv.org/pdf/2411.14257); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1c44e46358e0fb94dc94ec495a7fb1a-Paper-Conference.pdf)
- **Kang et al., "Unfamiliar Finetuning Examples Control How Language Models Hallucinate" (ICML 2024 / NAACL 2025, arXiv 2403.05612).**
  - As inputs become more unfamiliar, outputs default to a "hedged" prediction. Its form is set by how the *unfamiliar* fine-tuning examples were supervised: it minimizes aggregate loss over those examples.
  - Controlled SFT, RL and reward-model experiments on TriviaQA and MMLU.
  - Changing the supervision on unfamiliar examples (for example, to "I don't know") controls predictions on unfamiliar inputs.
  - (e) Learning-dynamics mechanism.
  - [arXiv 2403.05612](https://arxiv.org/pdf/2403.05612)
- **Liu, Liu, Sun, Hu, "Answering the Unanswerable Is to Err Knowingly" (AAAI 2026, arXiv 2508.18760).**
  - Large reasoning models "possess sufficient cognitive capabilities to recognize the flaws" in unanswerable math questions (SUM dataset) but fail to abstain, revealing "a **misalignment between their internal cognition and external response**".
  - Fix: cognitive monitoring plus inference-time intervention, which raises abstention "while maintaining the reasoning performance".
  - Covers R1-Distill and Qwen3 families.
  - [arXiv 2508.18760](https://arxiv.org/pdf/2508.18760); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40496)
- **Gu et al., "Bridging the Detection-to-Abstention Gap in Reasoning Models under Insufficient Information" (arXiv 2605.28070).**
  - Names the **detection-to-abstention gap**: models detect that a problem is under-specified but still produce unsupported answers.
  - Proposes Judge-Then-Solve, which makes the model commit explicitly to an answerability judgment before solving. It reports Abstention@Detection near saturation.
  - [arXiv 2605.28070](https://arxiv.org/abs/2605.28070)
- **Wang et al., "Surgical, Cheap, and Flexible: Mitigating False Refusal via Single Vector Ablation" (arXiv 2410.03415).** In the *safety* over-refusal setting, a false-refusal vector is extracted and ablated from the residual stream so the model refuses benign-but-harmful-looking prompts less. The method works directly on the weights. This builds on Arditi et al. (2024), who showed that refusal is mediated by a single direction. — [arXiv 2410.03415](https://arxiv.org/pdf/2410.03415); [arXiv 2406.11717](https://arxiv.org/pdf/2406.11717)
- **Frank, "Detection Is Cheap, Routing Is Learned" (arXiv 2603.18280).**
  - Argues that alignment works at a *routing* layer between concept detection and behavioral policy.
  - Probe accuracy alone is non-diagnostic: null controls also reach 100%.
  - Ablating a sensitivity direction removes censorship in most of the nine models tested.
  - A conceptual analogue (detection vs. routing), not abstention on unanswerable questions.
  - [arXiv 2603.18280](https://arxiv.org/abs/2603.18280)
- **Two 2026 mechanistic papers, titles and abstract only:**
  - "A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals" (arXiv 2609.00760) studies whether knowledge-based and safety-based refusal share a mechanism, under matched contrastive conditions.
  - "Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs" (arXiv 2603.27518).
  - I could not see their findings.
  - [arXiv 2609.00760](https://arxiv.org/pdf/2609.00760); [arXiv 2603.27518](https://arxiv.org/html/2603.27518)
- **Related safety over-refusal mitigation: DCR, "Discern Truth from Falsehood" (ICLR 2026, arXiv 2603.03323).**
  - Says prior mitigations (data augmentation, activation steering) trade off: reducing over-refusal degrades the rejection of genuinely harmful prompts.
  - Its explanation is that alignment fails to discriminate *truly* toxic from *superficially* toxic prompts.
  - Proposes a contrastive-refinement stage before alignment, with "theoretical and empirical" support.
  - [arXiv 2603.03323](https://arxiv.org/pdf/2603.03323)

### Inferences
- Taken together, the evidence supports a two-component picture:
  1. An answerability or knownness **judgment** that is linearly decodable (Slobodkin) and causally steerable (Ferrando).
  2. A **response policy** (refuse or answer) that fine-tuning attaches to that judgment, or to surface cues (Ling et al.), only loosely.
- Over-abstention on valid questions then has two candidate sources. Either the policy keys on cues correlated with the training "refuse" items rather than on the judgment (prompt-artifact evidence, CRaFT static conflict, Kang's "hedged default" for unfamiliar-looking inputs), or the labels themselves misstate the model's knowledge (CRaFT dynamic conflict).
- Ferrando's result that steering the "unknown entity" direction makes the model refuse questions about *known* entities is the most direct causal demonstration that a known-vs-unknown judgment drives refusal behavior.
- I found **no paper (2023–2026) that runs a full mechanistic study of how *abstention training itself* produces over-abstention on answerable questions.** By "full mechanistic study" I mean probes before and after training, showing the judgment stays intact while the response threshold or routing shifts. The closest pieces are:
  - CRaFT: a representation-space conflict hypothesis, backed by a data-filtering intervention.
  - Ling et al.: surface-pattern imitation, shown by perturbation.
  - Kang et al.: learning dynamics in controlled experiments.
  - Ferrando et al.: causal steering of knownness producing refusal, studied in existing models, not before and after abstention training.

### Gaps
- None of the mechanistic papers I found (Slobodkin, Ferrando, 2508.18760, 2605.28070) reports over-abstention **numbers** for answerable questions after abstention training. Most study under-abstention, or steer existing models.
- I could not check whether 2508.18760's "cognitive monitoring" uses linear probes on hidden states or monitors reasoning-trace text.
- I could not access the findings of 2609.00760 and 2603.27518, which may hold the most direct mechanistic evidence. They are worth reading in full.
- I did not verify UnknownBench, SQuAD2-style "self-aware LLM" work (e.g., SelfAware, Yin et al. 2023) or other SQuAD2-style unanswerable-detection studies of how abstention generalizes to answerable questions; search budget went to the papers above.
