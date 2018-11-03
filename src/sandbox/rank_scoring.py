__author__ = 'David'

import numpy as np

raw_score = np.zeros((2,5))

raw_score[1][0] = 5
raw_score[1][1] = 2
raw_score[1][2] = 5
raw_score[1][3] = 12
raw_score[1][4] = 1

sorted_indices = np.argsort(raw_score[1,:])

print sorted_indices