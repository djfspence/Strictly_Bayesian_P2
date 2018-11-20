
import numpy as np
import sys

def gelman_rubin_diagnostic(input_array):

    '''Compute the Gelman-Rubin diagnostic R_hat'''

    #http://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf

    #takes an input array of m chains, n iterations, for p parameters [m,p,n]

    m = input_array.shape[0]
    p = input_array.shape[1]
    n = input_array.shape[2]

    print m, p, n

    #set up variables

    R_hat = np.zeros(p)




    theta_bar = np.mean(input_array, axis = 2)

    print 'theta_bar', theta_bar, theta_bar.shape

    sj2 = np.var(input_array, ddof=1, axis = 2, dtype=np.float64)

    print 'sj2', sj2, sj2.shape

    #shape in both cases in [chain, parameter]

    theta_2bar = np.mean(theta_bar, axis=0)

    #shape is simply [parameter]

    print 'theta_2bar', theta_2bar, theta_2bar.shape

    # B (n * sample variance of theta_bar over all chains)
    B = n * np.var(theta_bar, ddof=1, axis = 0, dtype=np.float64)

    print 'B', B, B.shape

    # W (mean of sample variances sj2 over all chains)
    W = np.mean(sj2, axis=0)

    print 'W', W, W.shape

    Var_theta = ((1.0 - (1.0/n))*W) + ((1.0/n)*B)

    print 'Var_theta', Var_theta, Var_theta.shape

    R_hat = np.sqrt(np.divide(Var_theta, W))

    print 'R_hat', R_hat, R_hat.shape

    return R_hat

#############
#
# Test

test_array = np.random.rand(5,6,1000)

print test_array

answer = gelman_rubin_diagnostic(test_array)

print 'answer', answer, answer.shape

