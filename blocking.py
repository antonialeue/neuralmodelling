import numpy as np
import matplotlib.pyplot as plt

## BLOCKING

s1 = np.ones(100)                                       # stimulus 1
s2 = np.concatenate( (np.zeros(50), np.ones(50)) )      # stimulus 2
rewards = np.ones(100)                                  # rewards

expectations_w1 = 1 - np.exp(-0.1 * np.arange(100))     # expectations of weights for stimulus 1
expectations_w2 = np.zeros(100)                         # expectations of weights for stimulus 2

w1 = np.empty(100)          # weight array for stimulus 1
w2 = np.empty(100)          # weight array for stimulus 2
w1[0], w2[0] = 0, 0         # initial weights

e = 0.1                     # learning rate

# learning loop
for i in range(1, len(s1)):
    v = w1[i-1] * s1[i] + w2[i-1] * s2[i]       # prediction
    d = rewards[i] - v                          # error

    # updating weights with Rescorla-Wagner rule
    w1[i] = w1[i-1] + e * d * s1[i]             
    w2[i] = w2[i-1] + e * d * s2[i]

# plotting
plt.plot(w1, color="blue", label="Learned weights of stimulus 1")
plt.plot(w2, color="red", label="Learned weights of stimulus 2")
plt.plot(expectations_w1, color="blue", linestyle="dotted", linewidth="4", label="Idealised weights of stimulus 1")
plt.plot(expectations_w2, color="red", linestyle="dotted", linewidth="4", label="Idealised weights of stimulus 2")
plt.title("Blocking: Idealised expectations and learned expectations")
plt.xlabel("Trials")
plt.ylabel("Weight")
plt.legend(loc="best")

plt.show()