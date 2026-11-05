# Chokepoint Shock Early Warning

An early-warning system that flags which French import flows (energy, fertilisers, basic chemicals, aluminium) are about to collapse when a shipping chokepoint is disrupted, and whether acting early pays off.

The models are trained on 2019–2025, a period that includes the Red Sea crisis. They are then tested on the Hormuz crisis of January–April 2026, which they never saw during training.

**Tools:** Python (pandas) for data preparation · Orange Data Mining 3.40 with the Text Mining add-on for modelling
**Data:** Eurostat Comext · IMF PortWatch · World Bank Pink Sheet · NY Fed GSCPI · GDELT

---

## Key results

| Engine | Question | Result on the Hormuz months |
|---|---|---|
| **Machine Learning** (Random Forest) | Will this flow's volume collapse in the next three months? | AUC 0.81. At a cost-based threshold of 0.21, 68 of 88 shocks are caught (recall 77%, precision 51%), against 51 for a no-AI volume rule |
| **Deep Learning** (neural network) | How large will the price change be? | Best test error is 25.7 points (NN_A) against 33.7 for the baseline. Every R² is negative, because no model anticipated a price level never seen in training |
| **NLP** (LDA topics, VADER sentiment) | Is the news turning before the ships stop? | The Hormuz oil & tanker topic rose from 13% to 25% of headlines in December–February, before the 28 February strikes. Sentiment turned only in March |

**Decision and value.** Act (safety stock or alternative supplier) when P(shock) ≥ 0.21.

- **Cost:** in the central cost scenario this costs EUR 1.25 million per EUR 100 million of imports at risk. That is 22% less than the no-AI rule and 42% less than doing nothing.
- **Protection:** it protects 70% of the import value hit by shocks, against 47% for the rule.
- **Cash:** it blocks a third of the cash that acting on every flow would need.

**Main limits.** There is one test crisis of four months. The cost rates are assumed. The data is market-level, not company data. Part of what the model detects is a collapse already under way. The price regression should not be used for crisis budgeting.

---

## Contents

1. [Business problem](#1-business-problem)
2. [Data sources](#2-data-sources)
3. [Data preparation and enrichment](#3-data-preparation-and-enrichment)
4. [Methodology](#4-methodology)
5. [Results — Machine Learning](#5-results--machine-learning)
6. [Results — Deep Learning](#6-results--deep-learning)
7. [Results — NLP](#7-results--nlp)
8. [Business value and KPIs](#8-business-value-and-kpis)
9. [Limitations](#9-limitations)
10. [Conclusion and next steps](#10-conclusion-and-next-steps)
11. [Repository structure](#11-repository-structure)
12. [References and data sources](#12-references-and-data-sources)

---

## 1. Business problem

**Framing.** The analysis is set up for the procurement team of a French importer of energy products, fertilisers, basic chemicals and aluminium. This is a typical buyer, not a named company. French national import statistics serve as the market benchmark such a buyer would use.

**Problem.** Two shipping chokepoints were disrupted within three years:

- **Red Sea, from December 2023.** Houthi attacks pushed ships away from the Red Sea. Transits through Bab el-Mandeb fell to about half their normal level, and Cape of Good Hope traffic rose by up to 66% (IMF PortWatch, monthly ratios to normal).
- **Hormuz, after the strikes of 28 February 2026.** The Strait of Hormuz almost closed. Ship transits fell to 4% of normal in March 2026 and stayed below 21% until October. Over the same weeks, Brent crude rose 65% and urea 85% on a three-month basis (World Bank Pink Sheet).

A buyer facing such an event has to decide quickly which product-origin flows to protect. France imports these 15 products from 39 relevant partner countries every month, which gives several hundred active flows at any time. Watching them by hand is not realistic, and acting on all of them ties up cash for nothing.

**Question.** For each product × supplier country flow, two things need answering:

- Will the import volume collapse in the next three months?
- Is it worth acting now (building safety stock or switching supplier) rather than waiting?

**Use.** The procurement team would run the system once a month when new customs data arrives, and immediately when a chokepoint slows down. The early-warning score ranks flows, and the decision rule says where acting pays.

**Design principle.** Every model is compared with a no-AI baseline. Training covers 2019–2025 and testing covers a crisis the models have not seen. If the approach works on that unseen crisis, it can be reused for the next one.

---

## 2. Data sources

Four public sources are combined. Eurostat Comext is the backbone: each row of the final table is one product imported by France from one partner country in one month. The other sources join on month, commodity or shipping route. All are free and official or research-grade. None contains personal or confidential data.

| # | Source | What it gives | Coverage used | Type | Role |
|---|---|---|---|---|---|
| 1 | **Eurostat Comext, DS-059322** | French imports by product (CN8) and partner: value (EUR) and quantity (kg) | 15 products, monthly Jan 2019–Aug 2026, 26,736 product × partner × month rows | Quantitative + qualitative (country, product, family) | Defines the target (a volume collapse) and the euro exposure behind the KPIs |
| 2 | **IMF PortWatch, Daily Chokepoints Data** | Daily ship transits through 28 chokepoints, by vessel type (satellite AIS) | Hormuz, Bab el-Mandeb, Suez, Cape of Good Hope, Bosporus; about 79,500 daily rows | Quantitative | Measures disruption directly, weeks before customs data shows it |
| 3 | **World Bank Pink Sheet + NY Fed GSCPI** | Monthly world prices (Brent, EU natural gas, urea, DAP, aluminium) and the Global Supply Chain Pressure Index | Monthly, 2019–2026 | Quantitative | Separates a world price rise from a route-specific shock; gives global supply-chain stress as context |
| 4 | **GDELT news** | Shipping and chokepoint news: article counts, tone, and 6,546 English headlines | Monthly counts 2019–2026; headlines Oct 2023–Sep 2026, 1,399 news domains | Qualitative (text) + quantitative (counts, tone) | Input for the NLP engine, and a test of whether news turns negative before ships disappear |

The World Bank and New York Fed series are counted as one "context" source because both describe the global market rather than a specific flow.

**Reliability and known limits**

- **Comext:** official customs data, but recent months are revised and some origins are hidden for confidentiality.
- **PortWatch:** counts are modelled from AIS signals and only include ships that broadcast them.
- **Pink Sheet:** values are revised, and series definitions change over time (urea is quoted f.o.b. Middle East since March 2022).
- **GSCPI:** one global number, so it varies little across flows.
- **GDELT:** collected automatically, so headlines include off-topic articles. Section 3.5 describes the cleaning.

Download dates are listed in [Section 12](#12-references-and-data-sources).

---

## 3. Data preparation and enrichment

Preparation was done in Python (pandas) on copies of the raw files in `Data/data_raw/`, which are kept unchanged. Orange reads two prepared files from `Data/data_clean/`:

- `analysis_table.csv`: 6,166 rows, 44 columns
- `headlines_clean.csv`: 6,546 headlines

### 3.1 From customs rows to eligible flows

1. **Remove aggregates and unknowns.** Eurostat totals (WORLD, intra-EU and euro-area aggregates) and confidential origin codes are dropped.
2. **Compute unit prices.** Unit price is value divided by quantity, in EUR per tonne. Months with zero quantity give no price.
3. **Build a complete monthly grid.** Each product × partner pair gets one row per month from January 2019, with zero where nothing was imported. The grid has 38,766 flow-months.
4. **Keep flows that matter to a buyer.** A flow-month is eligible when both conditions hold over the previous 12 months:
   - the average import value is at least EUR 1 million per month;
   - the flow was active in at least 10 of those months.

   Tiny or occasional flows give meaningless unit prices and volume swings. This step leaves 6,166 eligible flow-months covering 15 products and 39 partner countries.

### 3.2 The two targets

- **`supply_shock`** (Machine Learning target)
  - Equals 1 when the average monthly quantity over the next three months falls below 50% of the flow's average over the previous 12 months.
  - It marks a real supply collapse, not a normal fluctuation.
  - Shocks make up 18.8% of training rows and 29.4% of test rows.
- **`price_change_clip`** (Deep Learning target)
  - The percentage change between the unit price over the next three months and over the previous six months, both weighted by quantity (total value divided by total quantity).
  - Clipped at the 1st and 99th percentiles (−56% and +94%) so that a few extreme unit prices do not dominate the error.
  - 586 rows have no valid price (no imports in the future window) and are excluded from the regression.

**Why a volume-only shock.** A broader definition was tested first: price up more than 20% or quantity down more than 30%. It flagged 54% of eligible flow-months, mostly through ordinary price swings. The classification target was therefore tightened to a volume collapse below 50% of normal. This is rarer and closer to what hurts a buyer. Price risk is handled separately by the regression.

### 3.3 Enrichment: features joined to each flow

All features use only information available at the end of the month being scored. The future windows above are used only for the targets.

| Group | Features | Source |
|---|---|---|
| Flow history | `qty_vs_normal` and `qty3m_vs_normal` (this month and the last three months against the 12-month normal), `price_vs_normal_pct`, `log_base_value` (size of the flow), `active_12m` | Comext |
| Product and route | `family` (Energy, Fertilisers, Plastics & chemicals, Aluminium), `route` (Hormuz, Red Sea / Suez, Turkish Straits, no Middle East chokepoint, intra-EU), `partial_bypass`, `month_of_year` | Comext + route mapping |
| Chokepoint traffic | Monthly ship transits divided by each chokepoint's normal level, for all vessels and for tankers, at Hormuz, Bab el-Mandeb, Suez, Cape and Bosporus. `route_transit_ratio` gives the same ratio for the flow's own route | PortWatch |
| World prices and stress | One- and three-month changes in Brent, EU gas, urea, DAP and aluminium; the flow's own benchmark (`own_bench_chg3m`); GSCPI level and three-month change | Pink Sheet, GSCPI |
| News | `news_log` (log10 of the monthly article count), `news_surge` (count against recent normal) | GDELT |

The route is inferred from the partner's geography, so it is a constructed mapping, not an observed shipping record. For example:

- Imports from Qatar or Kuwait must pass Hormuz.
- Saudi Arabia and the UAE can partly bypass it through Red Sea or Gulf of Oman ports, hence `partial_bypass`.

### 3.4 Cleaning in Orange

- **Roles.** These columns are metas, never features:
  - the identifiers (`PARTNER`, `partner_short`, `PRODUCT`, `product_label`);
  - `date` and `set`;
  - the other task's target;
  - `news_tone`, which is missing for 3,739 rows because tone data starts in October 2023.
- **Missing values and scaling.** The Preprocess widget does three things:
  - imputes missing values (210 rows lack `price_vs_normal_pct`);
  - one-hot encodes categorical features;
  - standardises numeric features.

  It is fitted inside each learner on the training data only.
- **Train / test split by time.** The split is by date, never random, so no future information reaches the model.
  - `set = train`: December 2019 to December 2025 (5,867 rows).
  - `set = test` (Hormuz): January to April 2026 (299 rows).

### 3.5 Headlines for NLP

Headlines were pulled from GDELT for shipping, Red Sea, Hormuz and Gulf news. They were deduplicated and kept with their date, news domain and source country. Each headline is assigned to one of five periods:

- before the Red Sea crisis (from October 2023);
- the Red Sea crisis;
- the calm period 2024–26;
- the Hormuz crisis;
- after (August–September 2026).

---

## 4. Methodology

Each engine answers one question a buyer asks. All three were built in Orange Data Mining 3.40 with the Text Mining add-on. The workflows are `chokepoint_ML.ows`, `chokepoint_DL.ows` and `chokepoint_NLP.ows`.

| Engine | Buyer's question | Target / output | Models compared | Main metrics |
|---|---|---|---|---|
| Machine Learning | Which flows will collapse in the next three months? | `supply_shock` (0/1) | Constant, Logistic Regression, Random Forest | AUC, recall, precision, F1, confusion matrix |
| Deep Learning | How large will the price change be? | `price_change_clip` (%) | Constant, Linear and Ridge Regression, Random Forest, three neural networks | MAE, RMSE, R² |
| NLP | Is the news turning before the ships stop? | Topics and sentiment by month and period | LDA with 4–10 topics; VADER sentiment | Topic coherence, perplexity, readability; mean compound score |

### 4.1 Machine Learning: supply-shock classifier

- **Pipeline.** File → Select Rows (train / test) → Preprocess (impute, one-hot, standardise) → learners → Test and Score → Confusion Matrix, ROC Analysis, Predictions, Feature Importance. Rank scores the features on the training set.
- **Random Forest.** 200 trees, no depth limit, subsets smaller than 5 not split, replicable training. It handles mixed numeric and categorical features, captures interactions (for example route × traffic drop) and reports feature importance.
- **Logistic Regression.** L2-regularised. It is the interpretable linear benchmark.
- **Constant.** Predicts the majority class. It is the no-AI floor.
- **Evaluation.** 10-fold stratified cross-validation on the training set, then "Test on test data" on the Hormuz months.
  - Accuracy is not used to judge the models.
  - With 19–29% shocks, predicting "no shock" everywhere already scores 71–81%.
- **Decision threshold.** Orange's default cut-off of 0.5 is not tied to any cost.
  - Missing a shock costs the buyer roughly 3 to 10 times more than a false alarm (Section 8).
  - Cost-weighted ROC curves were examined at FN:FP cost ratios of 3:1, 5:1 and 10:1.
  - A Formula widget turns the Random Forest probability into an alert: `Alert = 1 if P(shock) ≥ 0.21`.
  - A threshold of 0.17 was also tested.
- **Class balancing.** Random Forest and Logistic Regression were run with and without "balance class distribution". The unbalanced models were kept and the threshold was moved instead, because threshold choice is explicit and tied to cost.

![Cost-weighted ROC analysis at a 5:1 FN:FP cost ratio](Figures/ML_workflow/ROC%20Analysis%20at%205,1%20ratio.png)

### 4.2 Deep Learning: price-change regression

The same file and split are used. Rows without a valid price are removed, which leaves 5,329 training rows and 251 test rows.

**Neural networks** (Orange multilayer perceptron, Adam solver, replicable):

| Network | Hidden layers | Activation | Max iterations | L2 regularisation |
|---|---|---|---|---|
| NN_A | 100 | ReLU | 200 | Low (near default) |
| NN_B | 50, 20 | ReLU | 1,000 | Medium |
| NN_C | 20 | tanh | 1,000 | High |

- **Benchmarks.** The networks only earn their place if they beat these:
  - Constant (predicts the training mean);
  - Linear Regression;
  - Ridge Regression (stronger regularisation);
  - Random Forest, with the same settings as in 4.1.
- **Evaluation.** 10-fold cross-validation on train, then the Hormuz test months. MAPE and sMAPE are not reported because many target values are close to zero, which inflates percentage errors.

### 4.3 NLP: topics and sentiment in the news

- **Corpus.** `title` as the text field, English.
- **Preprocess Text.** In order:
  - lowercase, and remove accents and URLs;
  - default tokenizer, then WordNet lemmatiser;
  - remove English stopwords plus a custom list of 28 generic news words ("say", "amid", "latest", "week"…), stored in `Data/data_clean/custom_stopwords.txt`;
  - remove number-only tokens;
  - keep words that appear in at least 5 headlines.
- **Topic Modelling.** LDA, compared at k = 4, 5, 6, 8 and 10.
- **Sentiment Analysis.** VADER, which suits short English news text and returns positive, negative, neutral and compound scores.
- **Aggregation.** Group By on period and on month (mean topic weights, mean and median compound score, counts), saved to CSV. Box Plot and Bar Plot are used for comparison.

### 4.4 Alternatives considered and rejected

| Alternative | Reason for rejection |
|---|---|
| Random train/test split | Months are correlated, so a random split would leak future information and overstate performance. Cross-validation is reported only as an optimistic reference. |
| Price +20% or volume −30% as the shock | Flagged 54% of flow-months and mixed ordinary price swings with supply problems. |
| Accuracy as the selection metric | Misleading for a rare event (see 4.1). |
| Deeper or larger neural networks | NN_B and NN_C already did worse than NN_A on test and in CV. More capacity adds overfitting risk on about 5,300 rows. |
| LSTM / time-series deep learning | About six years of monthly data per flow and only two crises in that window give too little history per sequence. |
| Sentiment with the Liu Hu lexicon | VADER handles negation and intensity ("not safe", "very tense"), which matter in headlines. |

---

## 5. Results — Machine Learning

The Random Forest ranks shock risk well on the Hormuz months it never saw (AUC 0.81). With the cost-based threshold of 0.21 it catches 68 of the 88 shocks (recall 77%), and half of its alerts are real shocks.

### 5.1 Model comparison

| Model | Setting | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| Constant | CV, train (5,867) | 0.500 | 0.812 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | CV, train | 0.763 | 0.811 | 0.247 | 0.493 | 0.165 |
| Random Forest | CV, train | **0.814** | 0.834 | 0.410 | 0.617 | 0.307 |
| Constant | Test, Hormuz (299) | 0.500 | 0.706 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | Test, Hormuz | 0.792 | 0.702 | 0.212 | 0.480 | 0.136 |
| Random Forest | Test, Hormuz | **0.810** | 0.732 | 0.245 | 0.722 | 0.148 |

*Precision, recall and F1 are for the shock class at Orange's default 0.5 cut-off.*

- **Random Forest is the best model in both settings.** In cross-validation, Orange estimates a 99.9% probability that its AUC is higher than Logistic Regression's.
- **AUC holds up on the new crisis.**
  - It moves from 0.814 in CV to 0.810 on Hormuz.
  - The ranking of flows by risk transfers from one crisis to the next, even though the two crises hit different goods: energy and fertilisers at Hormuz, container goods from Asia in the Red Sea.
- **The default cut-off is useless for the buyer.**
  - At 0.5, the Random Forest raises only 18 alerts and catches 13 of 88 shocks (recall 15%).
  - The probabilities are well ordered but low, because shocks were rare (19%) in training.
  - This is why the threshold must be set from costs rather than left at 0.5.

### 5.2 Choosing the alert threshold

| Policy on the Hormuz months | Alerts | Shocks caught (of 88) | False alarms | Missed shocks | Recall | Precision |
|---|---|---|---|---|---|---|
| Random Forest, threshold 0.5 | 18 | 13 | 5 | 75 | 15% | 72% |
| **Random Forest, threshold 0.21** | 134 | **68** | 66 | **20** | **77%** | 51% |
| Random Forest, threshold 0.17 | 169 | 74 | 95 | 14 | 84% | 44% |
| No-AI rule: act when this month's volume is below 50% of normal | 90 | 51 | 39 | 37 | 58% | 57% |

- **Threshold 0.21.** The model finds 68 of 88 shocks. Lowering the threshold to 0.17 adds 6 more shocks but 29 more false alarms.
- **Against the no-AI rule** (react to a volume drop already visible in this month's customs data):
  - the model catches 17 more shocks (68 against 51) and misses 17 fewer;
  - it also raises 27 more false alarms;
  - Section 8 shows that this trade pays off in euros, because a missed shock costs far more than a false alarm.

![Random Forest confusion matrix on the Hormuz months at threshold 0.21](Figures/ML_workflow/Hormuz%20confusion%20matrix%20at%20threshold%200.21.png)

**Where the alerts fall (threshold 0.21)**

| Month (2026) | Shocks | Caught | False alarms |
|---|---|---|---|
| January | 18 | 15 | 16 |
| February | 19 | 14 | 14 |
| March | 20 | 18 | 21 |
| April | 31 | 21 | 15 |

| Product family | Flows | Shocks | Caught | False alarms |
|---|---|---|---|---|
| Energy | 154 | 60 | 51 | 47 |
| Fertilisers | 61 | 20 | 12 | 11 |
| Aluminium | 16 | 7 | 5 | 2 |
| Plastics & chemicals | 68 | 1 | 0 | 6 |

- **Before the strikes.** The model already flagged most January and February shocks, before the strikes of 28 February. Those flows were weakening in volume before the closure, and the model reads that.
- **April** is the weakest month (21 of 31). The shock spread then to flows whose recent volumes still looked normal.
- **By family.** Energy carries most alerts and most catches. Plastics and chemicals were almost untouched, and the model kept alerts there low.

### 5.3 What drives the prediction

- **Training set** (Rank widget, Information Gain Ratio and Gini):
  - the top features are `qty_vs_normal`, `qty3m_vs_normal`, `active_12m` and `price_vs_normal_pct`;
  - ReliefF also puts `family` and `route` high.
- **Hormuz test set** (Feature Importance, permutation, 15 repeats, decrease in AUC):
  - `qty_vs_normal` dominates, with about 0.085 AUC lost when shuffled;
  - next come `qty3m_vs_normal`, `price_vs_normal_pct` and `route`, at about 0.025–0.03 each;
  - then `active_12m`, `family`, `route_transit_ratio` and `log_base_value`.

![Permutation feature importance on the Hormuz test set](Figures/ML_workflow/Feature_importance_Test-set.png)

**Reading.** The strongest signal is the flow's own recent behaviour: a flow already below normal volume is the most likely to collapse. Route and the traffic ratio add the chokepoint context. World prices, GSCPI and news counts barely move the AUC on their own. They are the same for every flow in a given month, so they cannot separate flows within a month. They matter for timing, not ranking.

**Limitation.** Because recent volume is the top driver, part of what the model detects is a collapse that has already started. It is still useful to the buyer: it turns a weak signal across hundreds of flows into a ranked alert list, and it caught 17 more shocks than the volume rule alone.

---

## 6. Results — Deep Learning

No model predicts the size of the Hormuz price jump reliably. The best neural network (NN_A) has the lowest test error, 25.7 points against 33.7 for the baseline. However, every R² on the test months is negative, and the network's small lead over Random Forest is not consistent.

![Mean absolute error by model: cross-validation vs Hormuz test](Figures/report/dl_mae_comparison.png)

*Orange Test and Score, `chokepoint_DL.ows` · train n = 5,329, Hormuz test n = 251*

| Model | CV RMSE | CV R² | Test RMSE | Test R² |
|---|---|---|---|---|
| Constant | 26.09 | 0.000 | 39.63 | −1.552 |
| Linear Regression | 19.27 | 0.454 | 34.59 | −0.943 |
| Ridge Regression | 19.27 | 0.454 | 34.57 | −0.941 |
| Random Forest | 13.92 | 0.715 | 33.94 | −0.872 |
| **NN_A** | 14.87 | 0.675 | **32.46** | **−0.711** |
| NN_B | 15.36 | 0.653 | 36.14 | −1.122 |
| NN_C | 15.65 | 0.640 | 36.92 | −1.215 |

### 6.1 Interpretation

- **Random cross-validation is too optimistic.**
  - Random Forest's MAE rises from 9.0 in CV to 27.2 on test.
  - Neighbouring months of the same flow are very similar, so random folds share information between training and validation.
  - The Hormuz test is the honest estimate.
- **A level shift the models never saw.**
  - The average actual price change on the test months is +34.6%, against +3.7% in training.
  - A model trained on 2019–2025 has no example of such a level, so all R² values are negative: every model does worse than simply predicting the test months' own average.
  - On MAE, all models still beat the Constant baseline, by 11% (NN_C) to 24% (NN_A).
- **Regularisation adds nothing.** Ridge is practically identical to Linear Regression.
- **The neural network is not convincingly worth it.**
  - In cross-validation, Random Forest clearly beats NN_A (Orange estimates a 98.5% probability).
  - On the test months, NN_A beats Random Forest by 1.5 MAE points, on a single period of 251 rows.
  - NN_B and NN_C do worse than Random Forest. More layers (NN_B) or a smaller, more regularised network (NN_C) did not help.

### 6.2 Where the network goes wrong: timing

| Month (2026) | Flows | Mean actual change | Mean NN_A prediction | Error |
|---|---|---|---|---|
| January | 63 | +19.5% | +9.0% | −10.5 |
| February | 67 | +34.6% | +3.4% | −31.2 |
| March | 65 | +42.1% | +67.0% | +24.9 |
| April | 56 | +43.0% | +43.8% | +0.7 |
| **Total** | **251** | **+34.6%** | **+30.3%** | **−4.3** |

- **The network reacts about one month late.**
  - Prices for the next three months were already rising in February, but NN_A predicted almost no change.
  - It jumps in March, after the 28 February strikes, overshoots, and catches up in April.
  - The small overall bias (−4.3) hides large monthly errors in opposite directions.
- **Predictions form bands by month.** This is visible in the scatter plot of predicted against actual values (correlation 0.29).

  ![NN_A predicted vs actual price change on the Hormuz test months, coloured by month](Figures/DL_workflow/%28NN_A%29%20predicted%20vs%20actual%20price%20change%20on%20the%20Hormuz%20test%20months,%20coloured%20by%20month.png)
- **Timing is what fails, not separation.** Within a month the network does separate families: in March it predicts +28.5% for aluminium against +75.1% for plastics and chemicals.

The other models are no better at timing. No model anticipates the January–February rise. Mean predictions by month (January / February / March / April):

| Model | Jan | Feb | Mar | Apr | Note |
|---|---|---|---|---|---|
| Random Forest | 8.1 | 4.2 | 9.3 | 50.3 | Only reacts in April |
| Linear Regression | 1.2 | −8.2 | 54.0 | 38.0 | Reacts in March |

**Consequence.** The regression cannot forecast the size of product-level price jumps in an unprecedented crisis. It is kept as supporting evidence: it confirms the direction once the shock is visible. Prices rose in every family on the test months, and energy had the largest average rise (+38.1%). The early-warning decision relies on the classifier in Section 5.

**Note on the target.** `price_change_clip` looks three months ahead. For January and February 2026, that window already includes March, so part of what the model was asked to predict happened after the strikes. This is intended, since it is the buyer's real horizon, but it makes the January–February months hard by construction.

---

## 7. Results — NLP

The news gives an early signal through its topics, not its tone. The share of headlines about Hormuz oil and tankers doubled between December 2025 and February 2026, before the strikes. Sentiment only turned clearly negative in March, after them.

### 7.1 Five topics in the shipping news

LDA with k = 5 was retained:

- Coherence was highest at k = 8, but only 0.008 above k = 5, which is within LDA's run-to-run variation.
- Perplexity rises with k.
- At k = 5 every topic is distinct and readable. At k = 8 topics start to repeat.
- The k = 5 model has a coherence of 0.365.

| Topic | Label | Top keywords |
|---|---|---|
| 1 | Iran–US/Israel tension | iran, trump, gulf, war, oman, military, israel, persian |
| 2 | Hormuz oil & tanker | hormuz, iran, strait, oil, war, price, tanker, reopen |
| 3 | Houthi attacks | yemen, attack, houthi, ship, sea, red, strike, missile |
| 4 | Gulf diplomacy & routes | saudi, oil, global, iran, talk, arabia, shipping, route |
| 5 | Red Sea rerouting | sea, red, shipping, attack, ship, houthi, maersk, canal, suez |

**Mean topic weight by period**

| Period | Headlines | T1 Iran tension | T2 Hormuz oil | T3 Houthi | T4 Gulf diplomacy | T5 Red Sea rerouting |
|---|---|---|---|---|---|---|
| Before Red Sea (Oct–Nov 2023) | 257 | 0.21 | 0.21 | 0.17 | 0.21 | 0.19 |
| Red Sea crisis (Nov 2023–Jun 2024) | 2,219 | 0.09 | 0.12 | **0.39** | 0.11 | **0.30** |
| Calm (Aug 2024–Feb 2026) | 2,390 | 0.21 | 0.16 | 0.27 | 0.20 | 0.16 |
| Hormuz crisis (Feb–Jul 2026) | 1,443 | 0.15 | **0.31** | 0.18 | 0.21 | 0.15 |
| After (Aug–Sep 2026) | 237 | 0.12 | 0.23 | 0.26 | 0.24 | 0.14 |

The topics track the crises as expected. Houthi attacks and rerouting dominate the Red Sea period, and Hormuz oil and tankers dominate the Hormuz period. This confirms that the topic model captures real content, not noise.

### 7.2 Topic and sentiment over time

![Hormuz topic share and VADER sentiment by month](Figures/report/nlp_topic_sentiment_timeline.png)

*Orange Group By outputs, `output/sentiment_by_month_NLP.csv` and `output/topics_by_month_NLP.csv` · 34 months with headlines (Jul 2024 and Feb 2025 missing)*

| Period | Headlines | Mean compound | Median compound | Mean negative |
|---|---|---|---|---|
| Before Red Sea | 257 | −0.155 | 0.000 | 0.144 |
| Red Sea crisis | 2,219 | −0.330 | −0.422 | 0.220 |
| Calm 2024–26 | 2,390 | −0.144 | 0.000 | 0.140 |
| Hormuz crisis | 1,443 | −0.265 | −0.318 | 0.187 |
| After | 237 | −0.205 | −0.250 | 0.159 |

- **Both crises are clearly negative.**
  - Mean compound sentiment falls from about −0.15 in calm periods to −0.33 (Red Sea) and −0.27 (Hormuz).
  - In calm periods the median headline is neutral (0). In crises it is negative.
- **Red Sea: news led traffic by one to two months.**
  - Houthi-attack headlines jumped to 58% of the topic weight in November 2023.
  - Sentiment fell to −0.39 in December.
  - Bab el-Mandeb transits only fell to 53% of normal in January 2024 (PortWatch).
- **Hormuz: the topic led, the tone did not.**
  - Topic 2 rose from 13% in December 2025 to 21% in January and 25% in February, as tension built.
  - Sentiment stayed mild (−0.06 to −0.12) until March (−0.33).
  - Headline volume also jumped, from 126 in January to 396 in February.
- **False signals exist.** Sentiment dropped to −0.43 in July 2025 with no collapse in chokepoint traffic. Tone alone would have raised a false alarm.

**Use for the buyer.** A rise in the Hormuz topic share and in headline volume is a cheap, early prompt to review exposed flows. It is not precise enough to trigger action by itself. In the ML model, the news enters only as monthly article counts (`news_log`, `news_surge`). These are the same for every flow and do not rank flows (Section 5.3). Adding the topic share as a feature is a next step.

---

## 8. Business value and KPIs

Acting on the Random Forest alerts is the cheapest policy in every cost scenario tested. In the central scenario it costs EUR 1.25 million per EUR 100 million of imports at risk. The no-AI volume rule costs EUR 1.60 million and doing nothing costs EUR 2.13 million. That is a saving of 22% against the rule and 42% against inaction.

### 8.1 Decision rule

For each flow, the buyer acts only when the expected cost of a shock is higher than the cost of acting. Acting means building safety stock or switching to a supplier on another route. With the Random Forest this becomes: **act when P(shock) ≥ 0.21**. Otherwise the buyer keeps monitoring and no cash is blocked.

### 8.2 Cost assumptions

All figures are scenarios built on public data. They are not a client's accounts.

- **Exposure.** Three months of the flow's normal import value (its 12-month average monthly value × 3), from Comext.
- **Cost of acting** (false alarm or true alert). 1.75% of the exposure, the middle of a 1.5–2% range.
  - This covers carrying extra safety stock for a few months (financing and storage).
  - The stock is used later, so only the carrying cost is lost.
- **Cost of a missed shock.** Extra landed cost plus disruption.
  - In training data, flows hit by a supply shock saw their price rise 5.7 points more over three months than other flows (+8.8% against +3.1%). This gives 6% of exposure as the low case.
  - Shortages that stop production (energy or fertiliser inputs) cost more, so 10% and 17.5% are also tested.
  - These put a missed shock at about 3.4, 5.7 and 10 times the cost of a false alarm. This matches the 3:1 to 10:1 range used to set the threshold.
- **Simplification.** Acting is assumed to avoid the shock cost entirely. In practice a buffer may only cover part of it, which would reduce the gains of every acting policy.

### 8.3 Results on the Hormuz months (299 flow-months, January–April 2026)

The three cost columns use the three scenarios for the cost of a missed shock, in EUR million per EUR 100 million at risk.

| Policy | Flows acted on | Cost: shock = 3.4× action | Cost: central, 5.7× | Cost: high, 10× | Cash blocked in safety stock | Shock exposure protected |
|---|---|---|---|---|---|---|
| Never act (no AI) | 0 | 1.28 | 2.13 | 3.73 | EUR 0 | 0% |
| Act on every flow | 299 | 1.75 | 1.75 | 1.75 | EUR 13.2 bn | 100% |
| No-AI rule: volume below 50% of normal | 90 | 1.14 | 1.60 | 2.45 | EUR 3.4 bn | 47% |
| **Random Forest, P(shock) ≥ 0.21** | 134 | **0.99** | **1.25** | **1.72** | EUR 4.6 bn | **70%** |

*Cash blocked assumes one month of normal imports held as extra stock per flow acted on. It is a national total across all French importers of these flows, so a single client would hold its own share.*

### 8.4 The three KPIs

- **Margin (landed cost avoided).** The AI policy has the lowest expected cost in all three scenarios. In the central case it saves EUR 0.35 million per EUR 100 million at risk against the no-AI rule, and EUR 0.89 million against doing nothing.
- **Cashflow (cash tied in safety stock).**
  - The AI policy blocks EUR 4.6 bn of stock across the market, a third of what acting on every flow would need (EUR 13.2 bn).
  - It blocks EUR 1.2 bn more than the volume rule. That extra cash buys 23 more points of protected exposure.
- **Sales (supply secured).**
  - The AI policy protects 70% of the import value hit by a shock (EUR 5.9 bn of EUR 8.4 bn), against 47% for the volume rule.
  - Without a client's sales margin this cannot be converted into revenue, so it is reported as supply value secured.

**Where acting on everything wins.** At the high cost ratio (10:1), acting on every flow (1.75) is about as cheap as the AI policy (1.72), but it blocks almost three times more cash. The AI policy is the only option that is cheap on margin and moderate on cash in every scenario.

**Scaling to a client.** Take a buyer importing EUR 100 million per quarter of these products through exposed routes. In the central case and under these assumptions, that buyer would expect to save about EUR 350,000 a quarter in a crisis like Hormuz by using the model instead of the volume rule.

---

## 9. Limitations

- **One test crisis, four months.** The test set has 299 flow-months and 88 shocks from a single event. AUC 0.81 and the cost gains are estimates with real uncertainty, and a different crisis could give different numbers.
- **Threshold chosen with knowledge of the test period.**
  - The cost-weighted ROC analysis was run on training cross-validation, but the 0.21 and 0.17 thresholds were also compared on the Hormuz months.
  - A fully clean protocol would fix the threshold on training data only and report the test once.
- **The model partly detects collapses already under way.** The top driver is the flow's current volume against normal. The early-warning value comes from ranking hundreds of flows at once, not from predicting events with no visible sign.
- **Market data, not a company's books.**
  - Comext gives national imports, and a real client's prices, contracts and stock levels would differ.
  - The euro KPIs rely on assumed cost rates (Section 8.2). These are tested in three scenarios but not calibrated on a company.
- **Constructed route mapping.** The chokepoint each flow passes through is inferred from the partner's geography, not observed. Some flows, for example from Saudi Arabia or the UAE, can partly bypass Hormuz.
- **Satellite and news data are estimates.**
  - PortWatch counts only ships broadcasting AIS.
  - GDELT collects headlines automatically and includes off-topic articles.
  - The GDELT extract has no headlines for July 2024 and February 2025.
- **Feature outliers.** `price_vs_normal_pct` contains extreme values (up to +5,297%) from tiny or unusual shipments. They were standardised but not clipped. Random Forest is robust to this, but the linear models and neural networks may be affected.
- **Overlapping targets.** Both targets look three months ahead, so consecutive months of the same flow share part of their future window. This is why random cross-validation is optimistic and why the time-based test is the reference.
- **Data revisions.** Recent Comext and Pink Sheet months are revised after publication. The extract dates from 16 October 2026, and later revisions may change a few values.

---

## 10. Conclusion and next steps

A model trained on 2019–2025 ranked French import flows by supply-shock risk on the Hormuz crisis it had never seen, with an AUC of 0.81. With a cost-based threshold, it caught 77% of the shocks. It also gave the lowest expected cost of all the policies tested, ahead of a no-AI rule and of acting on everything. That is the value of AI here: a ranked alert list across hundreds of flows that pays for its false alarms.

The three engines play different roles:

- **Machine Learning is the decision tool:** which flows to protect now.
- **Deep Learning did not forecast the size of the price jump.**
  - No model anticipated an increase never seen in training.
  - The neural network was not clearly better than Random Forest.
  - The practical lesson is that crisis budgets should not rely on point forecasts of crisis prices.
- **NLP gives the earliest, cheapest signal.**
  - The Hormuz topic share rose in January–February 2026, before the strikes, while sentiment only turned afterwards.
  - It is a prompt to review exposure, not a trigger.

**Next steps**

1. Fix the alert threshold on training data only and re-test once, to remove any selection on the test months.
2. Add the monthly Hormuz topic share and headline volume by route as ML features, so that news can rank flows and not just time them.
3. Re-run the test on May–August 2026 as Comext data arrives, to check whether the model holds as the crisis continues.
4. Calibrate the cost rates with a real buyer's contracts, stock and margin data, and convert supply secured into protected sales.
5. Build a sequence model (LSTM) on chokepoint traffic once more crisis history is available.

---

## 11. Repository structure

```
chokepoint-early-warning/
├── README.md
├── Workflows/
│   ├── chokepoint_ML.ows              # supply-shock classifier
│   ├── chokepoint_DL.ows              # price-change regression
│   └── chokepoint_NLP.ows             # topics and sentiment
├── Data/
│   ├── data_raw/                      # original downloads, unchanged
│   │   ├── Daily_Chokepoints_Data.csv          # IMF PortWatch, daily transits
│   │   ├── PortWatch_chokepoints_database.csv  # IMF PortWatch, chokepoint reference table
│   │   ├── data-19348868.csv                   # Eurostat Comext extract
│   │   ├── CMO-Historical-Data-Monthly.xlsx    # World Bank Pink Sheet
│   │   └── gscpi_data.xls                      # NY Fed GSCPI
│   └── data_clean/                    # prepared files
│       ├── analysis_table.csv         # ML / DL input (6,166 rows)
│       ├── headlines_clean.csv        # NLP input (6,546 headlines)
│       ├── custom_stopwords.txt       # extra stopwords for Preprocess Text
│       ├── comext_monthly_flows.csv
│       ├── flows_with_target.csv
│       ├── portwatch_monthly_ratios.csv
│       ├── gdelt_news_daily.csv
│       └── gdelt_news_monthly.csv
├── output/                            # saved Orange outputs
│   ├── ML_predictions_hormuz.csv      # Random Forest probabilities, Hormuz months
│   ├── DL_Predictions_hormuz.csv      # price-change predictions, Hormuz months
│   ├── sentiment_by_month_NLP.csv
│   ├── Sentiment_by_period_NLP.csv
│   ├── topics_by_month_NLP.csv
│   └── topics_by_period_NLP.csv
└── Figures/
    ├── report/                        # summary charts used in this README
    ├── ML_workflow/                   # ROC, confusion matrices, feature importance, rank
    ├── DL_workflow/                   # Test and Score, pivots, predicted vs actual
    └── NLP_workflow/                  # LDA k = 4–10, topic pivots, sentiment plots, word cloud
```

**To reproduce**

1. Open the `.ows` files in `Workflows/` with Orange Data Mining 3.40. The NLP workflow also requires the Text Mining add-on.
2. If Orange cannot find a file, open the File widget and select the matching file in `Data/data_clean/`. In the NLP workflow, also select `custom_stopwords.txt` again in Preprocess Text.

---

## 12. References and data sources

**Data**

- Eurostat. *International trade in goods*, dataset DS-059322 (EU trade since 2002 by HS2-4-6 and CN8). Extracted 16 October 2026. [ec.europa.eu/eurostat](https://ec.europa.eu/eurostat)
- IMF PortWatch. *Daily Chokepoints Data* and *Chokepoints* reference table. UN Global Platform and IMF. Downloaded 15 October 2026. [portwatch.imf.org](https://portwatch.imf.org)
- World Bank. *Commodity Price Data (The Pink Sheet)*, monthly, October 2026 edition. [worldbank.org/commodities](https://www.worldbank.org/en/research/commodity-markets)
- Federal Reserve Bank of New York. *Global Supply Chain Pressure Index (GSCPI)*. Downloaded 15 October 2026. [newyorkfed.org/research/gscpi](https://www.newyorkfed.org/research/policy/gscpi)
- The GDELT Project. News articles and tone on shipping and chokepoints, October 2023 – September 2026. [gdeltproject.org](https://www.gdeltproject.org)

**Methods and tools**

- Arslanalp, S., Koepke, R. and Verschuur, J. (2021). *Tracking Trade from Space: An Application to Pacific Island Countries*. IMF Working Paper 2021/225.
- Hutto, C. J. and Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *Proceedings of ICWSM*.
- Blei, D. M., Ng, A. Y. and Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Demšar, J. et al. (2013). Orange: Data Mining Toolbox in Python. *Journal of Machine Learning Research*, 14, 2349–2353.
