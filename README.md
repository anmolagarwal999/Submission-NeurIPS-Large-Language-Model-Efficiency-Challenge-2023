# Submission for the NeurIPS Large Language Model Efficiency Challenge

**Members:** Anmol Agarwal, Ajinkya Deshpande, Shashank Shet, Arun Iyer, Suresh Parthasarathy

**Affiliation:** Microsoft Research India

EDIT: the current submission is ranked 2nd across all entries when evaluated on a combination of both the hidden-secret-eval tasks, and public HELM tasks.

Our datasets were derived from CNN-DM, MMLU, BigBench, TruthfulQA, BBQ, **ARC (by AllenAI)**, GSM-8k and **MathQA (by Hendrycks)**. We not only include data from the tasks mentioned in the sample_conf file, but also include other diverse tasks from BigBench.

In order to add robustness and fairness to our models, we also include special queries in our dataset which have been perturbed to measure robustness and fairness in the same way as HELM does perturbations.
Our initial findings suggested that models like Mistral are very sensitive to the sequence in which various options are presented, so we also shuffled the options in the query to introduce **option permutation invariance** which led to slight gains in performance.

We also use an ensemble of models. For each query, we classify into which sort of task does it belong to (fact-based knowledge based task OR reasoning based-task) using Regex. 

For any queries, please contact at either of the following emails: t-agarwalan@microsoft.com, anmolagarwal4453@gmail.com, ariy@microsoft.com
