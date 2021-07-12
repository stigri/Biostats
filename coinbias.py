import matplotlib.pyplot as plt
import numpy as np
import random as rdm
from scipy import stats

# define the bias of a weighted coin, unbiased = 0.5, biased > 0.5 or biased < 0.5
bias = 0.7
# number of coin flips
n_flips = 10000
# array with range of possible biases
range_bias = np.arange(0, 1.01, 0.01)
# print(range_bias)
# array initialized with p(bias_tails) = 1/101
p_bias = np.ones(len(range_bias))/len(range_bias)
# print(p_tails_bias)

# creates empty array of flips
flips = [0] * n_flips
# print(flips)

# creates random numbers between 0 and 1.
for n in range(n_flips):
    flip = rdm.random()
# if number equal or less than bias it is count as tails (0), otherwise as heads (1).
    if flip > bias:
        flip = 0
    else:
        flip = 1
# fill array with flip
    flips[n] = flip
print(flips)

# calculate p-value for null hypothesis h0 that coin is unbiased (p = 0.5)
# define max alpha error (Test decides that h0 is false when it is actually true)
alpha = 0.01
n_heads = np.count_nonzero(flips)
# print(n_heads)
p_value = stats.binom_test(n_heads, n_flips, p=0.5, alternative="two-sided")
print('P-Value = ', p_value)
if p_value < alpha:
    print('ho rejected')
else:
    print('h0 accepted')

# calculate prior, likelihood and evidence for all flips
for n in range(n_flips):
    p_prior = p_bias
    likelihood = np.multiply(np.power(range_bias, flips[n]), np.power(1 - range_bias, 1 - flips[n]))
    evidence = sum(np.multiply(likelihood, p_prior))

# calculate posterior
    posterior = np.divide(np.multiply(likelihood, p_prior), evidence)
    p_bias = posterior

# create figure dynamically
fig = plt.figure()
plt.plot(range_bias, p_bias)
plt.suptitle('')
if p_value > alpha:
    plt.title('Number of Flips = {0}, P-Value = {1:1.4f}'.format(n_flips, p_value))
else:
    plt.title('Number of Flips = {0}, P-Value < 0.01'.format(n_flips, p_value))

plt.xlabel('Bias = 0.7 (unbiased = 0.5)')
plt.ylabel('P(Bias | Number of Flips)')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()





