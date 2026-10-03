Run `python build_spider_dbs.py` before proceeding.

# Task 1:

* Decoder-only LLMs
    Qwen3-8b
    GPT-OSS-20b

* Reasoning LLMs
    DeepSeek-R1-Distill-Qwen-7B	

RQ1: How do (selected) LLms perform on cross-domain Text-to-SQL?

## Methodology:

In this report we evaluate the performance of decoder-only language models as well as reasoning LLMs on cross-domain SQL. Specifically, we evaluate the performance of decoder-only LLMs Qwen3-8B [needs citation] (non-MoE) and GPT-OSS-20B [needs citation] (3.6B active params MoE), as well as reasoning LLM Deepseek-R1-Distill-Qwen-14B (non-MoE) by downloading the NF4 quantized checkpoints from Huggingface and running the models locally on 1x NVIDIA GeForce RTX 4090 GPU. To maximize GPU utilization, we evaluate using batch sizes of 64.

[we get below data using slurm]
We use all 1034 samples provided in the evaluation data under the Spider dataset [citation needed]. Given an example, we prompt the LLM:
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

We evaluate the output using the same metrics as the Spider paper [citation needed]. These are:

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
Figure [n]: SQL query examples in 4 hardness levels. Credit: Spider [citation needed]

[todo: update with exact IDs used]

## Results: 

Below we report how Qwen3-0.6B, OLMo-2, and Deepseek-R1-Distill-Qwen3-1.5B perform on the Spider benchmark across all 3 metrics (Component Matching, Exact Matching Accuracy, Execution Accuracy). We calculate the values per problem difficulty as well as overall, and display them in the below tables.

All models are evaluated zero-shot with the same prompt on the 20 Spider dev databases. Aggregated across all difficulties and databases, performance is low for every model: execution accuracy is 0.121 to 0.209 and exact matching accuracy is 0.085 to 0.123. Execution accuracy exceeds exact matching accuracy for every model (for example 0.209 vs 0.099 for Qwen3-0.6B), so at least part of the executed-correct output is written differently from the gold SQL. No model is best everywhere: the ranking depends on the metric, the database and the problem difficulty.

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


Based on the above tables, it appears the larger reasoning model is not always the best model. DeepSeek-R1-Distill-Qwen-1.5B (reasoning, 1.5B) does not beat the smaller non-reasoning models Qwen3-0.6B (thinking off) and OLMo-2-1B-Instruct on most metrics:

* ⭐ **Component matching (table 1):** aggregated across all difficulties, DeepSeek has the lowest recall (0.174 vs 0.209 / 0.211) and F1 (0.275 vs 0.314 / 0.317). It does have the highest accuracy* aggregated across all difficulties (0.661 vs 0.628 / 0.640), and the best accuracy on medium (0.642). So the claim holds for recall and F1, not for accuracy. Since accuracy only scores components the model actually wrote, this is consistent with DeepSeek often producing no usable SQL (see below): when it writes a component it is often right, but it frequently writes none. OLMo is best on easy (0.813 / 0.378 / 0.516) and has the best F1 aggregated across all difficulties.
* **Exact matching (table 2):** DeepSeek is lowest aggregated across all difficulties (0.085 vs 0.099 Qwen, 0.123 OLMo).
* **Execution accuracy (table 3):** DeepSeek is clearly lowest aggregated across all difficulties (0.121 vs 0.209 Qwen, 0.195 OLMo), and falls to 0.017 (hard) and 0.006 (extra).

**2. Performance varies widely across databases.** Tables 4 and 5 give execution and exact matching accuracy for each of the 20 unseen databases. Execution accuracy for Qwen3-0.6B ranges from 0.10 (course_teach) to 0.53 (singer), for OLMo-2-1B from 0.05 (pets_1) to 0.37 (singer), and for DeepSeek-R1-Distill-Qwen-1.5B from 0.00 (concert_singer) to 0.27 (singer, voter_1). The best model changes with the database: Qwen3-0.6B is highest on execution accuracy in 13 databases, OLMo-2-1B in 8 and DeepSeek-R1-Distill-Qwen-1.5B in 1 (ties counted for each tied model). For example, OLMo-2-1B is clearly best on car_1 (0.21 vs 0.11 and 0.05) and student_transcripts_tracking (0.24 vs 0.13 and 0.06), while Qwen3-0.6B is best on world_1, tvshow and orchestra. DeepSeek-R1-Distill-Qwen-1.5B has the highest exact matching accuracy on flight_2 (0.20 vs 0.07 and 0.10). Databases with few problems (real_estate_properties n=4, voter_1 n=15, battle_death n=16, museum_visit n=18) are too small to rank models reliably, since one problem changes a score by 6 to 25 percentage points.

**3. Reasoning costs far more tokens for no accuracy gain.** DeepSeek averages 851 generated tokens per problem aggregated across all difficulties (table 6) vs 87 for OLMo (about 10x) and 46 for Qwen (about 18x), and grows with difficulty (611 on easy to 1177 on extra). The non-reasoning models stay between 28 and 111 tokens on average.

**4. Between the two small models, there is no single winner.** OLMo is best on exact match aggregated across all difficulties (0.123) and on easy queries for every metric (e.g. execution 0.452 vs 0.210 Qwen). Qwen is best on execution accuracy aggregated across all difficulties (0.209) and on medium, hard and extra execution (0.276 / 0.126 / 0.114), where OLMo falls to 0.018 on extra. OLMo's exact-match edge is driven by easy problems.

**5. Performance falls with difficulty for every model**, mostly at hard and extra, where exact match is at most 0.034 and execution at most 0.126.

**Interpretation, and what it does not show.** Our working hypothesis is that Text-to-SQL at this scale does not benefit from an explicit reasoning trace, and that a 0.5B-1B non-reasoning model is sufficient. The data are consistent with this, but they do not establish that Text-to-SQL is "too simple" for reasoning models, because several confounds are not controlled:

* **Truncation and failed outputs for DeepSeek.** 98 of 1034 generations (9.5%) hit the 4096-token cap (vs 23 / 2.2% for Qwen and 9 / 0.9% for OLMo at their 256-token cap), and 87 DeepSeek predictions are fewer than 3 words, i.e. effectively no SQL. These count as failures on every metric and probably depress DeepSeek's recall, exact and execution scores. A larger token budget could change the ranking.
* **Model differences beyond reasoning.** The three models differ in size (0.6B / 1B / 1.5B), pretraining and post-training recipe, not only in reasoning. DeepSeek-R1-Distill-Qwen-1.5B is a distilled model, so "reasoning" and "distillation quality" are confounded.
* **Single run and sample size.** One seed, one run, no confidence intervals. Gaps of a few points (for example Qwen 0.099 vs OLMo 0.123 exact match, or Qwen 0.209 vs OLMo 0.195 execution) may not be significant.
* **Aggregated component matching.** The component row is our own macro-average (the mean of the 10 per-component accuracy and recall scores, with F1 computed from those two means), not a number Spider reports itself.


# Task 2:

RQ2: How do characteristics of reasoning traces (of reasoning LLM) relate to Text-to-SQL performance?

## Methodology

## Resources