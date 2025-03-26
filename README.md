# Text Style Transfer: TBD
Implement techniques in NLP that allow for the transformation of text from one style to another while retaining the original content’s meaning.

# Initial Proposal:
The paper [ParaDetox: Detoxification with Parallel Data](https://aclanthology.org/2022.acl-long.469.pdf) provides a pipeline for collecting parallel data for the **detoxification** task. 

This paper's contributors collect non-toxic paraphrases for over 10,000 English toxic sentences. This pipeline can distill a large corpus of paraphrases to get toxic-neutral sentence pairs. [The code is open-sourced](https://github.com/s-nlp/paradetox), which allows us to look through its mechanisms. The relevant dataset is also available on [Huggingface](https://huggingface.co/datasets/s-nlp/paradetox).

From the paper, models including but not only `CondBERT`, `ParaGeDi`, etc. have been tested. Therefore, for this project, we could potentially work on [T5 model](https://arxiv.org/abs/1910.10683), which has been used in the area of **text style**: [TextSETTR: Few-Shot Text Style Extraction and Tunable Targeted Restyling](https://arxiv.org/abs/2010.03802). [A code sample from an individual attempt](https://github.com/aoxolotl/TextSETTR) is available for reference.
