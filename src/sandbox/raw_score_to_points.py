import numpy as np
import sys

#max_competitors = 6

raw_scores = np.array([-2.0, np.nan, -3.0, np.nan, 7, 8,-4, -3, np.nan, np.nan, 20])

num_active_competitors = np.count_nonzero(~np.isnan(raw_scores))

#print 'num_active_competitors', num_active_competitors

argsort_raw_scores = np.argsort(-raw_scores)
sorted_raw_scores = -np.sort(-raw_scores)

#print 'argsort_raw_scores', argsort_raw_scores
#print 'sorted_raw_scores', sorted_raw_scores

point_scores_rank = np.zeros(raw_scores.shape[0])
point_scores_rank[:] = np.nan

point_scores = np.zeros(raw_scores.shape[0])
point_scores[:] = np.nan

#print 'point_scores_rank', point_scores_rank

point_scores_rank[0] = num_active_competitors

for rank in range(1, num_active_competitors):
    if sorted_raw_scores[rank] == sorted_raw_scores[rank-1]:
        point_scores_rank[rank] = point_scores_rank[rank-1]
    else:
        point_scores_rank[rank] = point_scores_rank[rank-1] - 1

#print 'point_scores_rank', point_scores_rank

point_scores[argsort_raw_scores] = point_scores_rank

print 'raw_scores', raw_scores
print 'point_scores', point_scores

# sys.exit()
#
# print 'raw_scores', raw_scores
#
# non_zero_raw_scores_idx = np.nonzero(raw_scores)[0]
#
# print 'non_zero_raw_scores_idx', non_zero_raw_scores_idx
#
# non_zero_raw_scores = raw_scores[non_zero_raw_scores_idx]
#
# print 'non_zero_raw_scores', non_zero_raw_scores
#
# argsort_non_zero_raw_scores = np.argsort(-non_zero_raw_scores)
#
# print 'argsort_non_zero_raw_scores', argsort_non_zero_raw_scores
#
# num_active_competitors = non_zero_raw_scores.shape[0]
#
# print 'num_active_competitors', num_active_competitors
#
# active_points = np.zeros(num_active_competitors)
#
# active_points[0] = num_active_competitors
#
# for i in range(1, num_active_competitors):
#     if non_zero_raw_scores[argsort_non_zero_raw_scores[i]] == non_zero_raw_scores[argsort_non_zero_raw_scores[i]-1]:
#         active_points[i] = active_points[i-1]
#     else:
#         active_points[i] = active_points[i-1] - 1
#
#     active_points_2 = active_points[argsort_non_zero_raw_scores]
#
# print 'active_points_2', active_points_2
#


