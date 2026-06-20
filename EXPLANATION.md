# EXPLANATION.md — Track B (ML Engineer )

> 🚀 **Live Streamlit Demo** — run `streamlit run streamlit_app.py` locally, or deploy to
> [Streamlit Cloud](https://share.streamlit.io) for a public link (see README for instructions).

## Everyone answers these

### 1. What percentage of customers have `y = yes`? What does this imbalance mean for evaluation?

**11.70%** (5,289 out of 45,211 customers). 88.3% never subscribed.

That imbalance means accuracy is a misleading headline number — a model
that predicts `"no"` for every single customer would score 88.3% accuracy
while being completely useless for the actual business goal (finding the
customers worth pitching to). It's why this project reports precision,
recall, and F1 for the `"yes"` class specifically, uses
`class_weight="balanced"` on all models so they're not just learning to
ignore the minority class, and uses a stratified train/test split so the
~12% rate is preserved in both sets.

### 2. Which job category had the highest subscription rate? Does this make intuitive sense?

**Student** had the highest rate at **28.68%**, followed by **retired**
at 22.79%. Both are well above the next group (unemployed, 15.5%) and far
above categories like blue-collar (7.27%) or entrepreneur (8.27%).

This makes sense to me once I stopped reading it as "students have more
money" (they don't — average balances for this group are low) and read it
instead as "students and retirees have fewer competing financial
products and more flexible time." Both groups are less likely to already
be juggling a mortgage, multiple loans, or a packed work schedule, which
removes two of the biggest friction points an RM runs into when pitching
a new product. It lines up with the housing-loan finding below: customers
*without* a housing loan subscribe at more than double the rate of those
with one (16.7% vs 7.7%), and students/retirees are exactly the segments
least likely to be carrying one.

## Track B questions

### 3. Which feature had the highest importance in your tree-based model? Why?

**Updated after ensemble training.** The deployed model is a Random Forest.
Raw feature importances from the model (top 10 one-hot columns):

| Feature (raw OHE column) | Importance |
|---|---|
| age | 0.2047 |
| balance | 0.1979 |
| housing_no | 0.1442 |
| housing_yes | 0.1423 |
| loan_yes | 0.0370 |
| marital_single | 0.0335 |
| marital_married | 0.0300 |
| loan_no | 0.0300 |
| job_blue-collar | 0.0281 |
| education_tertiary | 0.0237 |

Aggregating one-hot columns back to their original feature (so
`housing_no` + `housing_yes` count together, etc.):

| Feature | Aggregated Importance |
|---|---|
| housing (loan) | **0.2865** |
| age | 0.2047 |
| balance | 0.1979 |
| loan (personal) | 0.0670 |
| marital | 0.0635 |
| job | 0.0281+ |
| education | 0.0237+ |
| default | ~0.005 |

**Housing loan status** remains the top feature overall when properly
aggregated — but age and balance are now close seconds. That tracking
matches the raw EDA: customers without a housing loan subscribe at roughly
2.2× the rate of those with one. A mortgage is a large, fixed monthly
commitment that crowds out both the spare cash and the appetite for a new
financial product — so it acts as a strong, simple signal for "how much
room does this person have to take on something new?"

Age and balance climbing so close to housing makes sense too: both are
continuous signals the tree can split on repeatedly at different thresholds,
giving them more opportunities to accumulate importance than a binary flag
like housing.

*(Note: `duration` — the call-length column — is deliberately excluded from
this model. It's a documented leakage feature: known only after the RM has
already made the call. Including it would dominate every importance ranking
but make the model useless for pre-call decisions.)*

### 4. Why is F1 a better metric than accuracy for this dataset?

Because of the 88/12 imbalance described in Q1, accuracy rewards a model
for defaulting to the majority class. F1 is the harmonic mean of precision
and recall *for the class we actually care about* (`"yes"`), so a model
only scores well on F1 if it's both finding subscribers (recall) and not
drowning the RM team in false leads (precision). For a cross-sell tool,
both failure modes are real costs — missed customers are lost revenue,
and too many false positives waste RM time and erode trust in the tool —
so optimizing for F1 instead of accuracy keeps the model honest about
both.

### 5. Walk through one of your 5 sample predictions — do you agree with the model?

**Customer #18275**: age 41, management, single, secondary education,
balance €1,138, no default, no housing loan, no personal loan.
**Model predicted "yes"** at 55.6% probability. **Actual outcome: "no."**

I think the model's call was reasonable even though it was wrong. Every
signal it had access to points the right way: above-median balance, no
housing loan, no personal loan, a stable job category — on paper this is
exactly the kind of "has room to take on a new product" customer the
housing-loan finding above would predict should subscribe. But 55.6% is
barely past the decision threshold, not a confident call, and that's the
honest part: the model is telling us this is a near-coin-flip case, and the
coin landing the other way isn't a model failure. It's the irreducible part
of human behavior that profile data alone can't capture (was this customer
in the market for a term deposit *right now*, did they have other plans for
that balance, did they just dislike the offer?). A near-50/50 miss like this
one is the model behaving correctly given the limits of what it was fed.

---

## Appendix — Ensemble Learning Results

### Why ensemble learning?

The single-model baseline (Random Forest, F1 = 0.3238 in the original run)
leaves performance on the table. Ensemble methods attack this from two
complementary directions:

- **Bagging** (RF, Extra Trees): trains many trees on random data/feature
  subsets and averages their votes, reducing variance without increasing bias.
- **Boosting** (HistGradientBoosting): trains trees sequentially, each one
  correcting the residuals of the previous — reduces bias and is especially
  effective on imbalanced data when `class_weight="balanced"` is set.
- **Soft Voting**: averages `P(yes)` probabilities from RF + ExtraTrees +
  HistGB — smooths individual model overconfidence.
- **Stacking**: learns *how much to trust* each base model by training a
  Logistic Regression meta-learner on out-of-fold (cv=5) predictions —
  more adaptive than equal-weight voting.

### Final leaderboard (test set, F1 for 'yes' class)

| Model | Accuracy | Precision | Recall | **F1** |
|---|---|---|---|---|
| **Random Forest** ← saved | 0.7141 | 0.2223 | 0.5775 | **0.3210** |
| Stacking Ensemble | 0.6994 | 0.2161 | 0.5974 | 0.3174 |
| Voting Ensemble | 0.6988 | 0.2147 | 0.5926 | 0.3152 |
| Gradient Boosting (Hist) | 0.7009 | 0.2142 | 0.5832 | 0.3133 |
| Extra Trees | 0.6568 | 0.1933 | 0.6096 | 0.2936 |
| Logistic Regression | 0.6224 | 0.1800 | 0.6267 | 0.2797 |

### Key observations

1. **Random Forest still wins on F1**, but the gap to Stacking/Voting is
   narrow (~0.004). With more hyperparameter tuning or SMOTE resampling, the
   stacking approach is likely to overtake it.

2. **All ensemble methods beat Logistic Regression** — F1 improves ~15% from
   the weakest to the strongest model, validating the ensemble investment.

3. **Recall is consistently high** (0.58–0.63 across all models), which is
   the right priority for a cross-sell tool — finding actual subscribers
   matters more than perfect precision, provided false-positive rate stays
   manageable for RM workload.

4. **Precision is the binding constraint** (~0.18–0.22 across all models).
   This reflects the hard ceiling imposed by only 8 profile features and no
   campaign-history data. The next lever to pull for a production system
   would be richer feature engineering (recency of previous contact, product
   holding patterns, transaction velocity) — not more ensemble complexity.

5. **`HistGradientBoostingClassifier` was used** instead of the legacy
   `GradientBoostingClassifier` because it natively supports
   `class_weight="balanced"` (the old one silently ignores it, causing recall
   to collapse to near-zero on an 88/12 imbalanced dataset).
