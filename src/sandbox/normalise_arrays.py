import numpy as np

vector = np.array([-2.0, np.nan, -3.0, np.nan, 7, 8,-4, -3, np.nan, np.nan, 20])

print vector


print np.mean(vector)
print np.std(vector)

vector_non_nan = vector[~np.isnan(vector)]

print vector_non_nan

print np.mean(vector_non_nan)
print np.std(vector_non_nan)

normalised_vector = (vector - np.mean(vector_non_nan))/np.std(vector_non_nan)

print normalised_vector

norm_vector_non_nan = normalised_vector[~np.isnan(vector)]

print np.mean(norm_vector_non_nan)
print np.std(norm_vector_non_nan)

#(x - np.mean(x))/np.std(x)