Run `python build_spider_dbs.py` before proceeding.

# Task 1:

* Decoder-only LLMs
    Qwen3-0.6B
    OLMo-2-0425-1B-Instruct

* Reasoning LLMs
    DeepSeek-R1-Distill-Qwen-1.5B

RQ1: How do (selected) LLms perform on cross-domain Text-to-SQL?

## Methodology:

In this report we evaluate the performance of decoder-only language models as well as reasoning LLMs on cross-domain SQL. Specifically, we evaluate the performance of the decoder-only LLMs Qwen3-0.6B [2] (with thinking mode disabled) and OLMo-2-0425-1B-Instruct [3], as well as the reasoning LLM DeepSeek-R1-Distill-Qwen-1.5B [4], all of which are dense (non-MoE) models, by loading the bf16 checkpoints from Huggingface and running the models locally on 1x NVIDIA GeForce RTX 4090 GPU. To maximize GPU utilization, we evaluate using batch sizes of 128.

[we get below data using slurm]
We use all 1034 samples provided in the evaluation data under the Spider dataset [1]. Given an example, we prompt the LLM:
```
Given the following SQLite database schema, write a SQL query that answers
the question.

{schema}

Question: {question}

Respond with only the SQL query in a ```sql code block.
```

For example, one of the schemas used is::
```
CREATE TABLE "stadium" (
  "Stadium_ID" NUMERIC, "Location" TEXT, "Name" TEXT, "Capacity" NUMERIC,
  "Highest" NUMERIC, "Lowest" NUMERIC, "Average" NUMERIC,
  PRIMARY KEY ("Stadium_ID")
)

CREATE TABLE "singer" (
  "Singer_ID" NUMERIC, "Name" TEXT, "Country" TEXT, "Song_Name" TEXT,
  "Song_release_year" TEXT, "Age" NUMERIC, "Is_male" TEXT,
  PRIMARY KEY ("Singer_ID")
)

CREATE TABLE "concert" (
  "concert_ID" NUMERIC, "concert_Name" TEXT, "Theme" TEXT, "Stadium_ID" TEXT, "Year" TEXT,
  PRIMARY KEY ("concert_ID"),
  FOREIGN KEY ("Stadium_ID") REFERENCES "stadium"("Stadium_ID")
)

CREATE TABLE "singer_in_concert" (
  "concert_ID" NUMERIC, "Singer_ID" TEXT,
  PRIMARY KEY ("concert_ID"),
  FOREIGN KEY ("concert_ID") REFERENCES "concert"("concert_ID"),
  FOREIGN KEY ("Singer_ID") REFERENCES "singer"("Singer_ID")
)
```

we could ask the question:
```
How many singers do we have?
```

to form a prompt like:

```
Given the following SQLite database schema, write a SQL query that answers
the question.

CREATE TABLE "stadium" (
  "Stadium_ID" NUMERIC, "Location" TEXT, "Name" TEXT, "Capacity" NUMERIC,
  "Highest" NUMERIC, "Lowest" NUMERIC, "Average" NUMERIC,
  PRIMARY KEY ("Stadium_ID")
)

CREATE TABLE "singer" (
  "Singer_ID" NUMERIC, "Name" TEXT, "Country" TEXT, "Song_Name" TEXT,
  "Song_release_year" TEXT, "Age" NUMERIC, "Is_male" TEXT,
  PRIMARY KEY ("Singer_ID")
)

CREATE TABLE "concert" (
  "concert_ID" NUMERIC, "concert_Name" TEXT, "Theme" TEXT, "Stadium_ID" TEXT, "Year" TEXT,
  PRIMARY KEY ("concert_ID"),
  FOREIGN KEY ("Stadium_ID") REFERENCES "stadium"("Stadium_ID")
)

CREATE TABLE "singer_in_concert" (
  "concert_ID" NUMERIC, "Singer_ID" TEXT,
  PRIMARY KEY ("concert_ID"),
  FOREIGN KEY ("concert_ID") REFERENCES "concert"("concert_ID"),
  FOREIGN KEY ("Singer_ID") REFERENCES "singer"("Singer_ID")
)

Question: How many singers do we have?

Respond with only the SQL query in a ```sql code block.
```

For each sample, we decode the model's generated tokens into text using the model-specific tokenizer, and check for the presence of a `</think>` token. If `</think>` was generated, we take all text generated before `</think>` as the reasoning trace and all text to the right as the output SQL. 

During decode, each model is run with greedy decoding so that we can get consistent results between runs. Models may generate up to 4096 tokens in the response, including reasoning traces. 

To evaluate cross-domain performance, we additionally report results separately for each of the 20 databases in the Spider dev set: world_1 (120 problems), car_1 (92), cre_Doc_Template_Mgt (84), dog_kennels (82), flight_2 (80), student_transcripts_tracking (78), wta_1 (62), tvshow (62), network_1 (56), concert_singer (45), pets_1 (42), poker_player (40), orchestra (40), employee_hire_evaluation (38), course_teach (30), singer (30), museum_visit (18), battle_death (16), voter_1 (15), and real_estate_properties (4).

We evaluate the output using the same metrics as the Spider paper [1]. These are:

1. Component Matching
    * For each of the following components:
        * SELECT
        * WHERE
        * GROUP BY
        * ORDER BY
        * KEYWORDS (including all SQL keywords without column names and operators)
    * Decompose each component in the prediction and ground truth as bags of several sub-components and check whether or not these two sets of components match completely.
    * Checks accuracy, recall, and F1
2. Exact Matching Accuracy
    * We measure whether the generated query as a whole is equivalent to the last section
    * The generated query is correct only if all components are correct
    * Checks only accuracy
3. Execution Accuracy
    * A list of gold values for each question is given
    * We measure how many gold values the generated query extracts when executed
    * Checks only accuracy

If output is invalid or unparsable, then it may fail all 3 dimensions (component matching, exact matching, execution accuracy). This is reasonable as invalid/unparsable output cannot be applied to SQL and thus must be avoided and penalized heavily.

We also consider the SQL Hardness Criteria, where queries are assigned difficulties based on the number of SQL components, selections, and conditions, so that queries that contain more SQL keywords are considered harder.

![alt text](image.png)
Figure [n]: SQL query examples in 4 hardness levels. Credit: Spider [1]

[todo: update with exact IDs used]

## Results: 

Below we report how Qwen3-0.6B, OLMo-2, and Deepseek-R1-Distill-Qwen3-1.5B perform on the Spider benchmark across all 3 metrics (Component Matching, Exact Matching Accuracy, Execution Accuracy). We calculate the values per problem difficulty as well as overall, and display them in the below tables.

All models are evaluated zero-shot with the same prompt on the 20 databases of the Spider dev set. Aggregated across all difficulties and databases, we find that every model performs poorly, with execution accuracy between 0.121 and 0.209 and exact matching accuracy between 0.085 and 0.123. Execution accuracy is higher than exact matching accuracy for all three models (for example 0.209 vs. 0.099 for Qwen3-0.6B), which suggests that at least some of the queries that return the correct result are written differently from the gold SQL. No single model is best across metrics, databases, and difficulty levels.

#### 1. Component matching (accuracy / recall / F1)

| Model | easy | medium | hard | extra | all |
|---|---|---|---|---|---|
| Problems (n) | 248 | 446 | 174 | 166 | 1034 |
| Qwen3-0.6B | 0.675 / 0.308 / 0.423 | 0.582 / 0.235 / 0.334 | 0.540 / 0.141 / 0.223 | 0.578 / 0.160 / 0.250 | 0.628 / 0.209 / 0.314 |
| OLMo-2-0425-1B-Instruct | 0.813 / 0.378 / 0.516 | 0.497 / 0.194 / 0.279 | 0.654 / 0.179 / 0.281 | 0.441 / 0.126 / 0.196 | 0.640 / 0.211 / 0.317 |
| DeepSeek-R1-Distill-Qwen-1.5B | 0.566 / 0.288 / 0.382 | 0.642 / 0.185 / 0.287 | 0.543 / 0.113 / 0.187 | 0.389 / 0.104 / 0.164 | 0.661 / 0.174 / 0.275 |

#### 2. Exact matching accuracy

| Model | easy | medium | hard | extra | all |
|---|---|---|---|---|---|
| Problems (n) | 248 | 446 | 174 | 166 | 1034 |
| Qwen3-0.6B | 0.157 | 0.137 | 0.006 | 0.006 | 0.099 |
| OLMo-2-0425-1B-Instruct | 0.379 | 0.061 | 0.034 | 0.000 | 0.123 |
| DeepSeek-R1-Distill-Qwen-1.5B | 0.214 | 0.078 | 0.000 | 0.000 | 0.085 |

#### 3. Execution accuracy

| Model | easy | medium | hard | extra | all |
|---|---|---|---|---|---|
| Problems (n) | 248 | 446 | 174 | 166 | 1034 |
| Qwen3-0.6B | 0.210 | 0.276 | 0.126 | 0.114 | 0.209 |
| OLMo-2-0425-1B-Instruct | 0.452 | 0.152 | 0.109 | 0.018 | 0.195 |
| DeepSeek-R1-Distill-Qwen-1.5B | 0.258 | 0.128 | 0.017 | 0.006 | 0.121 |

#### 4. Execution accuracy by database (domain)

| Database | Problems (n) | Qwen3-0.6B | OLMo-2-0425-1B-Instruct | DeepSeek-R1-Distill-Qwen-1.5B |
|---|---|---|---|---|
| world_1 | 120 | 0.183 | 0.100 | 0.108 |
| car_1 | 92 | 0.109 | 0.207 | 0.054 |
| cre_Doc_Template_Mgt | 84 | 0.202 | 0.250 | 0.071 |
| dog_kennels | 82 | 0.159 | 0.098 | 0.085 |
| flight_2 | 80 | 0.263 | 0.300 | 0.250 |
| student_transcripts_tracking | 78 | 0.128 | 0.244 | 0.064 |
| wta_1 | 62 | 0.161 | 0.274 | 0.129 |
| tvshow | 62 | 0.339 | 0.226 | 0.129 |
| network_1 | 56 | 0.286 | 0.232 | 0.179 |
| concert_singer | 45 | 0.111 | 0.089 | 0.000 |
| pets_1 | 42 | 0.167 | 0.048 | 0.071 |
| poker_player | 40 | 0.250 | 0.175 | 0.225 |
| orchestra | 40 | 0.350 | 0.225 | 0.100 |
| employee_hire_evaluation | 38 | 0.105 | 0.158 | 0.132 |
| course_teach | 30 | 0.100 | 0.133 | 0.100 |
| singer | 30 | 0.533 | 0.367 | 0.267 |
| museum_visit | 18 | 0.278 | 0.278 | 0.222 |
| battle_death | 16 | 0.375 | 0.188 | 0.188 |
| voter_1 | 15 | 0.267 | 0.200 | 0.267 |
| real_estate_properties | 4 | 0.500 | 0.250 | 0.000 |

#### 5. Exact matching accuracy by database (domain)

| Database | Problems (n) | Qwen3-0.6B | OLMo-2-0425-1B-Instruct | DeepSeek-R1-Distill-Qwen-1.5B |
|---|---|---|---|---|
| world_1 | 120 | 0.067 | 0.050 | 0.025 |
| car_1 | 92 | 0.043 | 0.152 | 0.033 |
| cre_Doc_Template_Mgt | 84 | 0.071 | 0.143 | 0.071 |
| dog_kennels | 82 | 0.085 | 0.037 | 0.061 |
| flight_2 | 80 | 0.075 | 0.100 | 0.200 |
| student_transcripts_tracking | 78 | 0.090 | 0.128 | 0.064 |
| wta_1 | 62 | 0.065 | 0.226 | 0.129 |
| tvshow | 62 | 0.226 | 0.194 | 0.097 |
| network_1 | 56 | 0.125 | 0.214 | 0.179 |
| concert_singer | 45 | 0.022 | 0.022 | 0.000 |
| pets_1 | 42 | 0.000 | 0.048 | 0.024 |
| poker_player | 40 | 0.175 | 0.150 | 0.100 |
| orchestra | 40 | 0.225 | 0.125 | 0.100 |
| employee_hire_evaluation | 38 | 0.105 | 0.105 | 0.105 |
| course_teach | 30 | 0.033 | 0.100 | 0.067 |
| singer | 30 | 0.333 | 0.233 | 0.167 |
| museum_visit | 18 | 0.056 | 0.111 | 0.111 |
| battle_death | 16 | 0.250 | 0.125 | 0.062 |
| voter_1 | 15 | 0.133 | 0.200 | 0.200 |
| real_estate_properties | 4 | 0.000 | 0.250 | 0.000 |

#### 6. Average generated tokens per problem

| Model | easy | medium | hard | extra | all |
|---|---|---|---|---|---|
| Qwen3-0.6B | 27.5 | 44.7 | 53.7 | 69.9 | 46.1 |
| OLMo-2-0425-1B-Instruct | 57.2 | 89.9 | 97.8 | 111.4 | 86.9 |
| DeepSeek-R1-Distill-Qwen-1.5B | 610.8 | 768.4 | 1096.0 | 1176.9 | 851.3 |


Based on the above tables, it appears that the larger reasoning model is not always the best model. DeepSeek-R1-Distill-Qwen-1.5B (reasoning, 1.5B) does not outperform the smaller non-reasoning models Qwen3-0.6B (thinking off) and OLMo-2-1B-Instruct on most metrics:

* ⭐ **Component matching (table 1):** aggregated across all difficulties, DeepSeek has the lowest recall (0.174 vs. 0.209 and 0.211) and F1 (0.275 vs. 0.314 and 0.317), but the highest accuracy (0.661 vs. 0.628 and 0.640), and the highest accuracy on medium problems (0.642). The observation that the larger reasoning model is not best therefore holds for recall and F1 but not for accuracy. Since accuracy only scores components that the model actually wrote, this is consistent with DeepSeek often producing no usable SQL (see below): when it writes a component it is often correct, but it frequently writes none. OLMo has the best F1 aggregated across all difficulties, and performs best on easy problems (0.813 / 0.378 / 0.516).
* **Exact matching (table 2):** aggregated across all difficulties, DeepSeek is lowest (0.085 vs. 0.099 for Qwen and 0.123 for OLMo).
* **Execution accuracy (table 3):** aggregated across all difficulties, DeepSeek is clearly lowest (0.121 vs. 0.209 for Qwen and 0.195 for OLMo), and its accuracy falls to 0.017 on hard and 0.006 on extra problems.

**2. Performance varies across databases.** Tables 4 and 5 report execution and exact matching accuracy on each of the 20 databases, all of which are unseen by the models. Execution accuracy ranges from 0.10 (course_teach) to 0.53 (singer) for Qwen3-0.6B, from 0.05 (pets_1) to 0.37 (singer) for OLMo-2-1B, and from 0.00 (concert_singer) to 0.27 (singer and voter_1) for DeepSeek-R1-Distill-Qwen-1.5B. The best model also differs by database: Qwen3-0.6B has the highest execution accuracy on 13 databases, OLMo-2-1B on 8, and DeepSeek-R1-Distill-Qwen-1.5B on 1 (ties are counted for each tied model). For example, OLMo-2-1B performs best on car_1 (0.21 vs. 0.11 and 0.05) and student_transcripts_tracking (0.24 vs. 0.13 and 0.06), while Qwen3-0.6B performs best on world_1, tvshow, and orchestra. DeepSeek-R1-Distill-Qwen-1.5B has the highest exact matching accuracy on flight_2 (0.20 vs. 0.07 and 0.10). Databases with few problems (real_estate_properties with n=4, voter_1 with n=15, battle_death with n=16, and museum_visit with n=18) are too small to rank the models reliably, since a single problem changes a score by 6 to 25 percentage points.

**3. Reasoning requires many more tokens without improving accuracy.** Aggregated across all difficulties, DeepSeek generates 851 tokens per problem on average (table 6), compared to 87 for OLMo (about 10x) and 46 for Qwen (about 18x). The number of tokens it generates also increases with difficulty, from 611 on easy problems to 1177 on extra problems, whereas the non-reasoning models stay between 28 and 111 tokens on average.

**4. There is no clear winner between the two smaller models.** OLMo has the best exact matching accuracy aggregated across all difficulties (0.123) and performs best on easy problems for every metric (for example 0.452 vs. 0.210 execution accuracy for Qwen). Qwen has the best execution accuracy aggregated across all difficulties (0.209) and on medium, hard, and extra problems (0.276, 0.126, and 0.114), where OLMo drops to 0.018 on extra problems. The exact matching advantage of OLMo is driven by easy problems.

**5. Performance decreases with difficulty for every model.** The decrease is largest on hard and extra problems, where exact matching accuracy is at most 0.034 and execution accuracy is at most 0.126.

**Interpretation and limitations.** Our working hypothesis is that Text-to-SQL at this scale does not benefit from an explicit reasoning trace, and that a non-reasoning model with 0.5B to 1B parameters is sufficient. The results are consistent with this hypothesis, but they do not establish that Text-to-SQL is too simple for reasoning models, because several confounds are not controlled:

* **Truncation and failed outputs for DeepSeek.** 98 of 1034 generations (9.5%) reach the 4096-token limit, compared to 23 (2.2%) for Qwen and 9 (0.9%) for OLMo at their 256-token limit, and 87 DeepSeek predictions contain fewer than 3 words, i.e. effectively no SQL. These count as failures on every metric and likely lower DeepSeek's recall, exact matching, and execution scores. A larger token budget could change the ranking.
* **Model differences beyond reasoning.** The three models differ in size (0.6B, 1B, and 1.5B parameters) and in pretraining and post-training, not only in whether they reason. DeepSeek-R1-Distill-Qwen-1.5B is also a distilled model, so the effect of reasoning cannot be separated from the quality of distillation.
* **Single run and sample size.** We report one run with one seed and no confidence intervals. Differences of a few points (for example 0.099 vs. 0.123 exact matching accuracy for Qwen and OLMo, or 0.209 vs. 0.195 execution accuracy) may not be significant.
* **Aggregated component matching.** The component matching row in table 1 is our own macro-average, namely the mean of the 10 per-component accuracy and recall scores with F1 computed from these two means, and is not a number reported by Spider.


# Task 2:

RQ2: How do characteristics of reasoning traces (of reasoning LLM) relate to Text-to-SQL performance?

## Methodology

## Results

# Task 3:

RQ3: To what extent does [strategy] improve [model]?

## Methodology:

## Resutls

# Task 4:

TODO


## References

[1] Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3911-3921, Brussels, Belgium. https://aclanthology.org/D18-1425/ (arXiv:1809.08887)

[2] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388. https://arxiv.org/abs/2505.09388

[3] Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, et al. 2025. 2 OLMo 2 Furious. arXiv:2501.00656. https://arxiv.org/abs/2501.00656

[4] DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948. https://arxiv.org/abs/2501.12948
