import numpy as np
import math
import matplotlib.pyplot as plt



n = np.arange(100)
print(n)
probs = np.zeros(len(n))
found = False
for ni in range(len(n)):

    prob = 1 - (math.factorial(365) / (math.factorial((365-ni)) * 365**ni))
    if prob > .5 and not found:

        print(f"n = {ni}")
        found = True
    probs[ni] = prob


plt.plot(n, probs)
plt.show()


print(math.factorial(10)/(math.factorial(4)*math.factorial(5)))

