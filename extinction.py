import numpy as np
import matplotlib.pyplot as plt


### EXTINCTION EXERCISE

# conditioned stimuli
cs = np.ones(101)

# unconditioned stimuli
us = np.concatenate( (np.ones(50), np.zeros(50)) )      # only 100 because we don't know whether there is an US in the last trial

# real states
states = np.concatenate( (np.zeros(50), np.ones(50)) )

# belief of the states
belief_of_state = np.concatenate( (np.ones(51), np.zeros(49), np.full(1, 0.5)) )

# intervals between trials (in days)
time = np.concatenate( (np.full(49, 1/1440), np.ones(1), np.full(49, 1/1440), np.full(1, 30)) )


## Part 2 -----------------------------------------------------------------------------------------

# plotting
plt.plot(belief_of_state, color="blue", label="expectation of US after CS")
plt.title("Animal's expectation of receiving the US after the CS")
plt.xlabel("Trials")
plt.ylabel("Expectation")
plt.legend(loc="best")
plt.show()


## Part 3 -----------------------------------------------------------------------------------------

def simple_heuristic(previous_state, similarity, time):
    
    beliefs = np.empty(2)

    # determine the previous state as believed by the animal (the one with higher probability)
    if previous_state > (1-previous_state):
        prev_in_s1 = True
    else:
        prev_in_s1 = False

    # probability of being in state 0
    s0_p = prev_in_s1 * (1-similarity) + (1-prev_in_s1) * similarity
    # probability of being in state 1
    s1_p = prev_in_s1 * similarity + (1-prev_in_s1) * (1-similarity)

    # time-related uncertainty: with longer time intervals passed since the last trial, the probabilities of the states decay to 0.5
    s0_p = 0.5 + (s0_p - 0.5) * np.exp(-0.1 * time)
    s1_p = 0.5 + (s1_p - 0.5) * np.exp(-0.1 * time)

    # normalization
    beliefs[0] = s0_p / (s0_p + s1_p)
    beliefs[1] = s1_p / (s0_p + s1_p)

    return beliefs


beliefs_over_time = np.empty((101, 2))
# the animal believes to be in state 0 with a probability of 1 before the the first trial
beliefs_over_time[0] = np.array([1,0])

for i in range(1, 101):

    if i == 1:
        similarity = 1          # shortly before the 2nd trial we don't know the second to last US
    elif us[i-1] == us[i-2]:
        similarity = 1
    else:
        similarity = 0

    ## version for passing the actual previous state (0 or 1) to the heuristic
    #beliefs_over_time[i] = simple_heuristic(states[i-1], similarity, time[i-1])

    ## version for passing the previous state belief (values in between 0 and 1)
    # instead of the previous actual state, use the belief for being in state 1 (closer to 1 indicates state 1, closer to zero indicates state 0)
    beliefs_over_time[i] = simple_heuristic(beliefs_over_time[i-1][1], similarity, time[i-1])

# plotting
plt.plot(beliefs_over_time[:,0], color="red", label="Belief of being in state 0")
plt.plot(beliefs_over_time[:,1], color="blue", label="Belief of being in state 1")
plt.title("Belief of states over time")
plt.xlabel("Trials")
plt.ylabel("Belief")
plt.legend(loc="best")
plt.show()


## Part 4 -----------------------------------------------------------------------------------------

w_s0 = np.empty(100)            # association weights in state 0
w_s1 = np.empty(100)            # association weights in state 1

w_s0[0], w_s1[0] = 0, 0         # initially weights are zero

e = 0.1                         # learning rate

# learning loop with Rescorla-Wagner
for i in range(1, 100):
    v = w_s0[i-1] * beliefs_over_time[i-1][0] + w_s1[i-1] * beliefs_over_time[i-1][1]   # prediction
    d = us[i] - v                                                                       # error
    w_s0[i] = w_s0[i-1] + e * d * beliefs_over_time[i-1][0]                             # update
    w_s1[i] = w_s1[i-1] + e * d * beliefs_over_time[i-1][1]                             # update

# plotting
plt.plot(w_s0, color="red", label="Expectation of US in state 0")
plt.plot(w_s1, color="blue", label="Expectation of US in state 1")
plt.title("State-dependent weights over trials")
plt.xlabel("Trials")
plt.ylabel("Weight")
plt.legend(loc="best")
plt.show()
