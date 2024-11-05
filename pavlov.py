import numpy as np
import matplotlib.pyplot as plt

stimuli = np.ones(75 + 1)   # stimuli
rewards = np.ones(75 + 1)   # rewards

expectations = 1- np.exp(-0.1 * np.arange(76))

weights = np.empty(76)      # weight array
initial_weight = 0          # initial weight
e = 0.1                     # learning rate

#v = initial_weight * stimuli[0]
#d = rewards[0] - v
#weights[0] = initial_weight + e * d * stimuli[0]      # stimmt das das man da den stimulus multipliziert?

weights[0] = 0

for i in range(1, len(stimuli)):
    v = weights[i-1] * stimuli[i]
    d = rewards[i] - v
    weights[i] = weights[i-1] + e * d * stimuli[i]


plt.plot(expectations, color="blue", linestyle="dashed")
plt.plot(weights, color="blue")
plt.show()