# Findings

Test set: 272 races, 2014 to 2026-20 (round 20 of the 2026 season). Every prediction was made with a model that had only seen earlier races.

## Q1 + Q2. Models vs baselines, day by day

Retrained before every race. Log loss: lower is better.

|                                | Winner picked   | Podium picks right   | Top-10 picks right   |   Log loss (win) |   Log loss (podium) |
|:-------------------------------|:----------------|:---------------------|:---------------------|-----------------:|--------------------:|
| Thursday / Simple baseline     | 37%             | 52%                  | 72%                  |            0.141 |               0.288 |
| Thursday / Logistic regression | 45%             | 59%                  | 75%                  |            0.121 |               0.247 |
| Thursday / Gradient boosting   | 42%             | 60%                  | 75%                  |            0.123 |               0.247 |
| Friday / Simple baseline       | 46%             | 56%                  | 74%                  |            0.135 |               0.278 |
| Friday / Logistic regression   | 45%             | 62%                  | 76%                  |            0.117 |               0.24  |
| Friday / Gradient boosting     | 43%             | 60%                  | 75%                  |            0.118 |               0.241 |
| Saturday / Simple baseline     | 55%             | 69%                  | 76%                  |            0.112 |               0.231 |
| Saturday / Logistic regression | 50%             | 69%                  | 78%                  |            0.105 |               0.215 |
| Saturday / Gradient boosting   | 56%             | 68%                  | 78%                  |            0.101 |               0.21  |

## Q3. Does the model survive a rule reset?

Logistic regression, retrained before every race. Reset seasons in the test set: 2014, 2017, 2022, 2026. 'Gap' is reset minus stable, with a 90% bootstrap interval.

| Day | Rounds | Podium picks right: stable | reset | Gap (90% interval) | Races: stable / reset |
|---|---|---|---|---|---|
| Thursday | 1-4 | 74% | 54% | -20 pts (-29 to -10) | 36 / 16 |
| Thursday | 5+ | 60% | 52% | -8 pts (-13 to -2) | 155 / 65 |
| Friday | 1-4 | 75% | 60% | -15 pts (-27 to -2) | 36 / 16 |
| Friday | 5+ | 61% | 57% | -4 pts (-9 to +1) | 155 / 65 |
| Saturday | 1-4 | 79% | 71% | -8 pts (-19 to +3) | 36 / 16 |
| Saturday | 5+ | 68% | 65% | -3 pts (-8 to +3) | 155 / 65 |

Thursday podium picks by season (reset seasons: 2014, 2017, 2022, 2026):

|   season | Simple baseline   | Logistic regression   | Gradient boosting   | Logistic minus baseline   |
|---------:|:------------------|:----------------------|:--------------------|:--------------------------|
|     2014 | 49%               | 60%                   | 54%                 | +11 pts                   |
|     2015 | 49%               | 72%                   | 77%                 | +23 pts                   |
|     2016 | 49%               | 67%                   | 63%                 | +17 pts                   |
|     2017 | 53%               | 52%                   | 55%                 | -2 pts                    |
|     2018 | 56%               | 59%                   | 57%                 | +3 pts                    |
|     2019 | 54%               | 60%                   | 60%                 | +6 pts                    |
|     2020 | 65%               | 71%                   | 71%                 | +6 pts                    |
|     2021 | 50%               | 70%                   | 70%                 | +20 pts                   |
|     2022 | 56%               | 55%                   | 55%                 | -2 pts                    |
|     2023 | 55%               | 50%                   | 53%                 | -5 pts                    |
|     2024 | 47%               | 50%                   | 50%                 | +3 pts                    |
|     2025 | 60%               | 67%                   | 64%                 | +7 pts                    |
|     2026 | 40%               | 43%                   | 48%                 | +3 pts                    |

## Q4. Which features survive a reset?

Rounds 1-4 only, all seasons from 2008. Average rank correlation between the feature and the finishing order (1.0 = feature orders the field perfectly, 0 = useless). 56 stable-season races, 20 reset-season races.

|                                      |   Reset seasons |   Stable seasons |   Lost in a reset |
|:-------------------------------------|----------------:|-----------------:|------------------:|
| Grid slot                            |            0.61 |             0.65 |              0.03 |
| Qualifying pace (this weekend)       |            0.62 |             0.66 |              0.04 |
| Practice pace (this weekend)         |            0.54 |             0.6  |              0.06 |
| Driver: last 5 races                 |            0.41 |             0.56 |              0.16 |
| Team: qualifying, last 5 races       |            0.44 |             0.6  |              0.17 |
| Team: last 5 races                   |            0.42 |             0.59 |              0.17 |
| Driver: last season's average finish |            0.38 |             0.56 |              0.18 |
| Team: last season's average finish   |            0.38 |             0.58 |              0.2  |

## Q5. Frozen model vs retraining after every race

|                                         | Podium picks right: frozen   | Podium picks right: updating   |   Log loss (podium): frozen |   Log loss (podium): updating |
|:----------------------------------------|:-----------------------------|:-------------------------------|----------------------------:|------------------------------:|
| Reset / Thursday / Logistic regression  | 52%                          | 52%                            |                       0.253 |                         0.247 |
| Reset / Thursday / Gradient boosting    | 50%                          | 53%                            |                       0.255 |                         0.242 |
| Reset / Friday / Logistic regression    | 56%                          | 58%                            |                       0.24  |                         0.235 |
| Reset / Friday / Gradient boosting      | 57%                          | 56%                            |                       0.236 |                         0.231 |
| Reset / Saturday / Logistic regression  | 66%                          | 66%                            |                       0.217 |                         0.211 |
| Reset / Saturday / Gradient boosting    | 65%                          | 65%                            |                       0.209 |                         0.204 |
| Stable / Thursday / Logistic regression | 61%                          | 62%                            |                       0.249 |                         0.247 |
| Stable / Thursday / Gradient boosting   | 60%                          | 62%                            |                       0.258 |                         0.248 |
| Stable / Friday / Logistic regression   | 63%                          | 63%                            |                       0.247 |                         0.242 |
| Stable / Friday / Gradient boosting     | 61%                          | 62%                            |                       0.249 |                         0.245 |
| Stable / Saturday / Logistic regression | 70%                          | 70%                            |                       0.22  |                         0.217 |
| Stable / Saturday / Gradient boosting   | 71%                          | 70%                            |                       0.216 |                         0.213 |

## 2026 race by race (Logistic regression)

|   round | Thursday: winner   |   Thursday: podium picks right | Saturday: winner   |   Saturday: podium picks right |
|--------:|:-------------------|-------------------------------:|:-------------------|-------------------------------:|
|       1 | no                 |                              1 | yes                |                              2 |
|       2 | no                 |                              2 | no                 |                              2 |
|       3 | no                 |                              1 | no                 |                              2 |
|       4 | no                 |                              2 | yes                |                              1 |
|       5 | no                 |                              0 | no                 |                              1 |
|       6 | yes                |                              1 | yes                |                              2 |
|       7 | no                 |                              1 | no                 |                              2 |
|       8 | no                 |                              2 | no                 |                              1 |
|       9 | no                 |                              2 | no                 |                              2 |
|      10 | no                 |                              1 | yes                |                              2 |
|      11 | no                 |                              1 | yes                |                              1 |
|      12 | no                 |                              1 | yes                |                              3 |
|      13 | no                 |                              1 | no                 |                              1 |
|      14 | yes                |                              2 | yes                |                              2 |
|      15 | no                 |                              1 | yes                |                              1 |
|      16 | no                 |                              1 | yes                |                              3 |
|      17 | no                 |                              1 | yes                |                              3 |
|      18 | no                 |                              2 | no                 |                              3 |
|      19 | yes                |                              2 | yes                |                              1 |
|      20 | no                 |                              1 | no                 |                              2 |
