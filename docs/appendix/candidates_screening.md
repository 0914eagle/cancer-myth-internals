# Row-addable candidates — ICLR/ICML 2026 medical NLP

Papers that (a) have a **live** GitHub repo, (b) propose a method and compare against
baselines, so their main table takes an extra row. Backbone read from the repo itself.
SAE column = a public SAE suite exists for that exact backbone.

| # | Paper | Venue | Repo | ★ | Backbone | SAE suite | Data |
|--:|---|---|---|--:|---|---|---|
| 1 | [AnesSuite: A Comprehensive Benchmark and Dataset Suite for](https://openreview.net/forum?id=iKRQMeC7yO) | ICLR | [MiliLab/AnesSuite](https://github.com/MiliLab/AnesSuite) | 33 | llama-3.1-8b, llama3.1-8b | llama scope | public |
| 2 | [From Conflict to Consensus: Boosting Medical Reasoning via](https://openreview.net/forum?id=S7lpdz7NAW) | ICML | [NJU-RL/MA-RAG](https://github.com/NJU-RL/MA-RAG) | 17 | qwen3-8b | qwen-scope | public |
| 3 | [MedSIGHT: Towards Grounded Visual Comprehension in Medical](https://openreview.net/forum?id=a7mORMEWYX) | ICML | [Aofei-Chang/MedSIGHT](https://github.com/Aofei-Chang/MedSIGHT) | 12 | llava, llava.eval.chatbot | qwen-scope | public |
| 4 | [SP-Mind: An Autonomous Reasoning Agent for Spatial Proteom](https://openreview.net/forum?id=UJcB3XrffF) | ICML | [tomtommyyuan/spmind](https://github.com/tomtommyyuan/spmind) | 277 | — | — | public |
| 5 | [MedAgent-Pro: Towards Evidence-based Multi-modal Medical D](https://openreview.net/forum?id=ZOuU0udyA4) | ICLR | [jinlab-imvr/MedAgent-Pro](https://github.com/jinlab-imvr/MedAgent-Pro) | 195 | internvl_decider.cpython-310.pyc, internvl_decider.py | — | public |
| 6 | [SleepLM: Natural-Language Intelligence for Human Sleep](https://openreview.net/forum?id=9wpwfSJCp9) | ICML | [yang-ai-lab/SleepLM](https://github.com/yang-ai-lab/SleepLM) | 49 | — | — | public |
| 7 | [3DMedAgent: Unified Perception-to-Understanding for 3D Med](https://openreview.net/forum?id=TH6pLxCOQ3) | ICML | [jinlab-imvr/3DMedAgent](https://github.com/jinlab-imvr/3DMedAgent) | 36 | — | — | public |
| 8 | [Ophiuchus: Incentivizing Tool-augmented ''Think with Image](https://openreview.net/forum?id=coJqVkqb03) | ICML | [SII-zyj/Ophiuchus](https://github.com/SII-zyj/Ophiuchus) | 30 | qwen2.5-7b_rm.sh, qwen2.5-vl-7b | — | public |
| 9 | [Boosting Medical Visual Understanding From Multi-Granular ](https://openreview.net/forum?id=ccjukmExrB) | ICLR | [HUANGLIZI/MGLL](https://github.com/HUANGLIZI/MGLL) | 26 | — | — | public |
| 10 | [CerebraGloss: Instruction-Tuning a Large Vision-Language M](https://openreview.net/forum?id=Xi1jkajWi9) | ICLR | [iewug/CerebraGloss](https://github.com/iewug/CerebraGloss) | 26 | — | — | public |
| 11 | [OmniCT: Towards a Unified Slice-Volume LVLM for Comprehens](https://openreview.net/forum?id=nrZI64gTvC) | ICLR | [ZJU4HealthCare/OmniCT](https://github.com/ZJU4HealthCare/OmniCT) | 20 | qwen2.5-3b-instruct, qwen2.5-7b-instruct | — | public |
| 12 | [Med-Scout: Curing MLLMs' Geometric Blindness in Medical Pe](https://openreview.net/forum?id=iTkQLqa1Ha) | ICML | [HKUSTGZ-ML4Health-Lab/Med-Scout](https://github.com/HKUSTGZ-ML4Health-Lab/Med-Scout) | 18 | internvl.py, internvl3 | — | public |
| 13 | [LiveClin: A Live Clinical Benchmark without Leakage](https://openreview.net/forum?id=E0WSAugJ0j) | ICLR | [AQ-MedAI/LiveClin](https://github.com/AQ-MedAI/LiveClin) | 17 | qwen2.5-vl-7b-instruct | — | public |
| 14 | [MedCRP-CL: Continual Medical Image Segmentation via Bayesi](https://openreview.net/forum?id=v0DWbfP3b9) | ICML | [zygao930/MedCRP-CL](https://github.com/zygao930/MedCRP-CL) | 14 | — | — | public |
| 15 | [From Conversation to Query Execution: Benchmarking User an](https://openreview.net/forum?id=hLweUPBz7k) | ICLR | [glee4810/EHR-ChatQA](https://github.com/glee4810/EHR-ChatQA) | 9 | qwen3-32b | — | public |
| 16 | [Seizure-Semiology-Suite($S^3$): A Clinically Multimodal Da](https://openreview.net/forum?id=MyorUlHKVc) | ICML | [LinaZhangUCLA/SeizureSemiologySuite](https://github.com/LinaZhangUCLA/SeizureSemiologySuite) | 6 | internvl3.5, internvl3.5-38b | — | public |
| 17 | [MedScope: Incentivizing "Think with Videos" for Clinical R](https://openreview.net/forum?id=OOyPj8hdiM) | ICML | [SII-WenjieLisjtu/MedScope](https://github.com/SII-WenjieLisjtu/MedScope) | 6 | qwen2.5-vl-7b-instruct | — | public |
| 18 | [Time-Conditioned Foreseeing: An EHR-Specific Foundation Mo](https://openreview.net/forum?id=IalpB5Mzaz) | ICML | [Pusheen-cat/TCF_PFM](https://github.com/Pusheen-cat/TCF_PFM) | 5 | — | — | public |
| 19 | [SynerMedGen: Synergizing Medical Multimodal Understanding ](https://openreview.net/forum?id=Tyv61ZKb9s) | ICML | [piooip/SynerMedGen](https://github.com/piooip/SynerMedGen) | 5 | — | — | public |
| 20 | [Resp-Agent: An Agent-Based System for Multimodal Respirato](https://openreview.net/forum?id=ZkoojtEm3W) | ICLR | [zpforlove/Resp-Agent](https://github.com/zpforlove/Resp-Agent) | 3 | — | — | public |
| 21 | [Benchmarking the Scientific Mind: A Pathology-Derived Biom](https://openreview.net/forum?id=Pj1Z7dWm4v) | ICML | [UniverseOfUniverse/SORBE](https://github.com/UniverseOfUniverse/SORBE) | 2 | qwen3-235b-a22b-instruct, qwen3-235b-a22b-instruct. | — | public |
| 22 | [Dynamic Decision Learning: Test-Time Evolution for Abnorma](https://openreview.net/forum?id=YcccAbXbVK) | ICML | [compai-lab/2026-ICML-DDL](https://github.com/compai-lab/2026-ICML-DDL) | 1 | — | — | public |
