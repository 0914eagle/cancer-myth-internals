"""PreWoMe-style structured premise review (Kim et al., EMNLP 2023), shared by the medical
and general-domain runners so both use identical prompt text.

Structure follows PreWoMe §3: question -> presupposition list -> feedback and action
(working memory) -> final answer generated with that working memory. The prompt wording is
ours (zero-shot); the paper's few-shot exemplars are not reproduced, so results are
"PreWoMe-style", not a reproduction. The presupposition step deliberately has NO
scope-preservation instruction: this row is the existing method, not our modification.
"""

PRESUPPOSITIONS = (
    "List the presuppositions of the following question: the statements it takes for granted. "
    "Write one per line, each starting with '- '. Write '- None' if it takes nothing for granted.\n\n"
    "Question: {question}"
)
FEEDBACK_ACTION = (
    "Below are a question and its presuppositions. For each presupposition, say whether it is "
    "true, false, or cannot be verified, and briefly why. Then write one action describing how "
    "the answer should respond to the question given these findings.\n\n"
    "Question: {question}\n\nPresuppositions:\n{presuppositions}\n\n"
    "Use this format:\nFeedback:\n- <presupposition>: <true/false/cannot be verified> - <reason>\nAction: <how to answer>"
)
ANSWER = (
    "Answer the question. Use the working memory below, which lists the question's "
    "presuppositions, feedback on them, and the planned action.\n\n"
    "Question: {question}\n\nWorking memory:\nPresuppositions:\n{presuppositions}\n\n{feedback}"
)
PROMPTS = {"prewome_presuppositions": PRESUPPOSITIONS, "prewome_feedback": FEEDBACK_ACTION,
           "prewome_answer": ANSWER}
