# Chokepoint Shock Early Warning — explained simply

## The idea in one sentence

When a key shipping route is blocked, this project predicts which French imports are about to run short, so a buyer can protect them before it happens.

---

## The problem

A large share of the world's oil, gas, fertilisers and chemicals travels by ship through a few narrow sea passages, called **chokepoints**. Two of them were disrupted recently:

- **Red Sea (from December 2023):** attacks on ships forced many to take a much longer route around Africa.
- **Strait of Hormuz (from 28 February 2026):** after military strikes, almost no ships could pass. Traffic fell to 4% of normal.

When this happens, a company that imports these goods into France has hundreds of supplies to watch (each product from each country). It cannot check them all by hand, and protecting all of them would be too expensive.

**So the question is:** which supplies will collapse in the next three months, and is it worth acting now?

---

## The data

Four free, public sources were combined:

| Source | What it tells us |
|---|---|
| **Eurostat** (EU trade statistics) | How much of each product France imports, from which country, every month |
| **IMF PortWatch** (satellite ship tracking) | How many ships pass through each chokepoint every day |
| **World Bank and New York Fed** | World prices of oil, gas, fertilisers and aluminium, and a global supply-chain stress index |
| **GDELT** (global news database) | News headlines about shipping, the Red Sea and Hormuz |

Everything was joined into one table: **one row = one product from one country in one month.**

---

## How it was tested fairly

The models learned from data from **2019 to 2025**, which includes the Red Sea crisis.

They were then tested on **January–April 2026, the Hormuz crisis**, which they had never seen.

This is like studying past exams and then sitting a new one. If the model works on a crisis it never saw, it can be trusted for the next one.

Every model was also compared with a **simple rule without AI**: "act when this month's imports are already below half of normal".

---

## The three models

All three were built in **Orange Data Mining**, a visual tool where you connect blocks instead of writing code.

### 1. Machine Learning — "Will this supply collapse?"

- **What it does:** gives each supply a risk score between 0 and 1.
- **Model used:** Random Forest (many decision trees that vote together).
- **Result:** it found **68 of the 88 real collapses** during the Hormuz crisis. The no-AI rule found only 51.
- **Ranking quality:** AUC of 0.81. An AUC of 0.5 is random guessing and 1.0 is perfect.
- **What it looks at most:** whether the supply was already weakening in recent months, and which sea route it uses.

### 2. Deep Learning — "How much will prices go up?"

- **What it does:** tries to predict the price change over the next three months.
- **Model used:** neural networks, compared with simpler models.
- **Result:** it did **not** work well. Prices jumped by about 35% on average during Hormuz, far more than anything seen in training. The models reacted about one month too late.
- **Lesson:** do not trust price forecasts during a crisis that has never happened before.

### 3. NLP (text analysis) — "Is the news warning us?"

- **What it does:** reads 6,546 news headlines and finds:
  - the main **topics** (LDA topic modelling);
  - whether the tone is positive or negative (VADER **sentiment** analysis).
- **Result:** news about "Hormuz, oil and tankers" doubled in **January–February 2026, before the strikes**. The tone of the news only became negative afterwards, in March.
- **Lesson:** what the news talks about gives an early warning. How negative it sounds does not.

---

## What it means for a business

**The rule:** protect a supply (extra stock or another supplier) when its risk score is **0.21 or higher**.

The threshold is low on purpose: missing a real shortage costs much more than a false alarm.

Compared with the alternatives, on the Hormuz months:

| Strategy | Cost per €100M of imports at risk | Share of affected imports protected |
|---|---|---|
| Do nothing | €2.13M | 0% |
| Simple rule (no AI) | €1.60M | 47% |
| **This model** | **€1.25M** | **70%** |
| Protect everything | €1.75M | 100%, but ties up about 3× more cash |

In short, the model is the **cheapest option**: about 22% cheaper than the simple rule and 42% cheaper than doing nothing.

---

## Limits to keep in mind

- It was tested on **one crisis** of four months only.
- The costs are **estimates**, not real company accounts.
- The data covers **all of France**, not one specific company.
- Part of what the model detects is a collapse that has **already started**. Its strength is ranking hundreds of supplies quickly.
- The price model (Deep Learning) should **not** be used to plan crisis budgets.

---

## Words used in this project

| Term | Meaning |
|---|---|
| **Chokepoint** | A narrow sea passage many ships must use (Hormuz, Red Sea, Suez…) |
| **Supply shock** | Imports of a product from a country fall below half of normal over the next three months |
| **AUC** | How well a model ranks risky cases above safe ones (0.5 = random, 1 = perfect) |
| **Recall** | Share of real shocks the model caught |
| **Precision** | Share of alerts that were real shocks |
| **Threshold** | The risk score above which an alert is raised |
| **Random Forest** | Many decision trees that vote together |
| **Neural network** | A model loosely inspired by the brain, good at complex patterns but needing lots of data |
| **LDA** | A method that finds the main topics in a set of texts |
| **VADER** | A tool that scores how positive or negative a sentence sounds |

---

## Where to look next

- `README.md`: full technical description.
- `Chokepoint_Shock_Early_Warning_Report.pdf`: complete report.
- `Workflows/`: the three Orange workflows (open them in Orange Data Mining 3.40).
