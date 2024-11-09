
import numpy as np
import matplotlib.pyplot as plt
import GeneralizedLambda as gl

## Fixing random state for reproducibility
#np.random.seed(19680801)
#
#mu, sigma = 100, 15
#x = mu + sigma * np.random.randn(10000)
#
## the histogram of the data
#n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
#
#
#plt.xlabel('Std from mu')
#plt.ylabel('Probability')
#plt.title('Histogram of IQ')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
#plt.grid(True)
#plt.show()

Cases   = 100000
N       = [0,.1975,.1349,.1349] #Parameters for normal GLG
x       = gl.DrawGeneralizedLambda(N,Cases)

#n, bins, patches = plt.hist(x.x, 11, normed=1, facecolor='g', alpha=0.75)
n, bins, patches = plt.hist(x.x, 51,normed = 1)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(0, 1, r'$\mu=100,\ \sigma=15$')
plt.axis([-4, 4, 0, .5])
plt.grid(True)
plt.show()