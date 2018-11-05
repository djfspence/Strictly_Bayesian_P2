import numpy as np


# judge_score_weight

# sum of a and b
beta_width_factor = 10.0

judge_score_weight_min = 0.1
judge_score_weight_max = 0.8

judge_score_weight_span = judge_score_weight_max - judge_score_weight_min

judge_score_weight_live = 0.75


# mu = (judge_score_weight_live - judge_score_weight_min) / judge_score_weight_span
#
# print mu
#
# a = mu * beta_width_factor
# b = beta_width_factor - a

for i in range(50):

    mode = 1.0 * (judge_score_weight_live - judge_score_weight_min) / judge_score_weight_span
    a = (mode * (1.0 * beta_width_factor - 2.0)) + 1.0
    b = beta_width_factor - a
    mode_new = np.random.beta(a, b)

    judge_score_weight_live = (mode_new * judge_score_weight_span) + judge_score_weight_min
    print i, judge_score_weight_live



