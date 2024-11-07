import numpy as np
import matplotlib.pyplot as plt

## OVERSHADOWING

s1 = np.ones(100)       # stimulus 1
s2 = np.ones(100)       # stimulus 2
rewards = np.ones(100)  # rewards

expectations = 1 - np.exp(-0.2 * np.arange(100))    # idealised expectations of reward

w1 = np.empty(100)      # weight array for stimulus 1
w2 = np.empty(100)      # weight array for stimulus 2
w1[0], w2[0] = 0, 0     # initial weights

e = 0.1                 # learning rate

# learning loop
for i in range(1, len(s1)):
    v = w1[i-1] * s1[i] + w2[i-1] * s2[i]       # prediction
    d = rewards[i] - v                          # error

    # updating weights with Rescorla-Wagner
    w1[i] = w1[i-1] + e * d * s1[i]
    w2[i] = w2[i-1] + e * d * s2[i]

# plotting
plt.axvline(99, color="grey", linestyle="dashed")
plt.plot(w1, color="blue", label="Learned weights of stimulus 1 & 2")
plt.plot(expectations, color="limegreen", linestyle="dashed", linewidth="1", label="Idealised expectations of reward")
plt.plot(100, w1[-1], "ro", color="blue", label="Result (stimulus 1 & 2)")
plt.text(20, 0.1, "Training", color="grey")
plt.title("Overshadowing: Idealised expectation of reward and learned weights")
plt.xlabel("Trials")
plt.ylabel("Weight")
plt.legend(loc="lower right")

plt.show()