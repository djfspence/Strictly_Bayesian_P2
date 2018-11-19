from scipy import stats
import numpy as np

rvs1 = np.array([0.5388, 0.7391, 0.6316, 0.5257, 0.2604, 0.6282, 0.7511, 0.4765, 0.4172, 0.561, 0.4694, 0.4796, 0.7271, 0.5617, 0.567, 0.5622, 0.6423])
rvs2 = np.array([0.5000, 0.6726, 0.6106, 0.5747, 0.6077, 0.4248, 0.4753, 0.7306, 0.3832, 0.6673, 0.4468, 0.6892, 0.5848, 0.6022, 0.5049, 0.6260, 0.4513])

print "%.4f" % np.mean(rvs1), "%.4f" % np.mean(rvs2)

print stats.ttest_ind(rvs1,rvs2)
print stats.ttest_rel(rvs1,rvs2)

rvs_size = rvs1.shape[0]

window = 10

for i in range(0, rvs_size-window, 1):

    r1 = rvs1[i:i+window]
    r2 = rvs2[i:i+window]

    print i
    print r1
    print r2
    print "%.2f" % stats.ttest_ind(r1,r2)[1]
