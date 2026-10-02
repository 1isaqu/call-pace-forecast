# Glossary

Terms used in the code, the API and the documentation.

## Domain

| Term | Meaning |
|---|---|
| outreach attempt | One attempt to contact a debtor (a call, a message). One row of `contact_events`. |
| answered call | An attempt whose result is flagged `answered`. |
| right-party contact (CPC) | An attempt whose result is flagged both `answered` and `right_party`: the person reached is the right person. |
| agreement | A payment arrangement. One agreement has one or more installments. |
| generated agreement | An agreement whose status is in the configured "generated" set. It counts on the day it was generated even if it breaks later. |
| installment | One scheduled payment of an agreement. `installment_no = 0` is the first installment (down payment). |
| first installment | The down payment, `installment_no = 0`. |
| broken / auto-broken agreement | An agreement that stopped being honoured, manually or automatically after a grace period. |
| exception (pending approval) | An agreement waiting for an approval. Not part of the generated set by default. |
| portfolio | A group of debts from one origin. The label lives in `debt_attributes`. |
| product line | A broad business line. In the examples, `auto` and `consumer`. Stored as the `bucket`. |
| agent | A person who contacts debtors. Identified by an invented `agent_id`. |
| debtor | The person who owes. No personal identifier is ever modelled. |
| ticket | The typical amount of one event, for example the down payment of an agreement. |
| avg ticket | Total amount divided by number of agreements. |
| grace period | Days after a due date during which a payment still counts as on time. |

## Forecasting

| Term | Meaning |
|---|---|
| bucket | The identifier of a series (product line, queue, team). Part of the data contract. |
| band | One hour of the pace table: expected, actual, delta, status, cumulative. Also, with quantile models, `lower`, `upper` and `position`. |
| expected | What the model predicts an hour should produce. |
| actual | What the hour produced, known when it closes. |
| delta | `actual - expected`. |
| pace | The whole table of bands for the day. Intraday pace. |
| cumulative | Total so far inside the day. |
| cumulative lag | Cumulative before the hour being predicted. The leakage-safe feature. |
| first-window total | Sum of the first hours of the day (default two), zero until the window has closed. |
| projected close | Actual so far plus expected for the hours that have not closed. A point estimate. |
| share | Cumulative divided by the day's final total, at a given hour. |
| lookup | Phase 1. Median count per (bucket, hour) over a trailing window. |
| KNN | Phase 2. K-nearest-neighbours regression over time-of-day features. |
| quantile band | Phase 3. A lower and an upper quantile model next to a Poisson centre model. |
| post-processing | The rounding applied to every model output and every offline metric (`ceil` by default, clipped at zero). |
| covariate | An optional numeric column that is constant within a bucket-day, such as days since an event. |
| maturity cut-off | Number of most recent days ignored when fitting, because they may be incomplete. |
| regime shift | A lasting change in the level of volume. |
| routing | Choosing a model per bucket. |

## Evaluation

| Term | Meaning |
|---|---|
| walk-forward | Evaluation where each test block is predicted by a model fitted only on earlier days. |
| fold | One train/test split of the walk-forward protocol. |
| holdout | The last fold, used once to confirm and never to select. |
| MAE | Mean absolute error per hour. |
| directional accuracy | Share of hours where the model deviates from the baseline in the same direction as reality. |
| coverage | Share of actual values inside the quantile band. |
| nominal coverage | What the band claims: the upper quantile minus the lower quantile. |
| WAPE | Weighted absolute percentage error: total absolute error over total actual. |
| promotion | Deciding that a challenger replaces the baseline, via explicit criteria. |
| ablation | Removing or swapping a part of the model to see whether it mattered. |
| Monte Carlo scenario | One simulated future: growth rate, daily totals and noise drawn at random. |
| dispersion | The `r` of the negative binomial; variance is `mean + mean^2 / r`. Smaller means noisier. |
