import numpy as np
import matplotlib.pyplot as plt


# conditioned stimuli
cs = np.ones(101)

# unconditioned stimuli
us = np.concatenate( (np.ones(50), np.zeros(50)) ) # do we know the last reward?? we dont -> no 101st CS

# actual states
states = np.concatenate( (np.ones(50), np.zeros(50)) )

# belief of the states
belief_of_state = np.concatenate( (np.ones(51), np.zeros(49), np.full(1, 0.5)) ) #????

# intervals between trials (in days)
time = np.concatenate( (np.full(49, 1/1440), np.ones(1), np.full(49, 1/1440), np.full(1, 30)) )


## Part 2 -----------------------------------------------------------------------------------------

plt.plot(belief_of_state, color="blue", label="belief state")
plt.title("Belief of state")
plt.xlabel("Trials")
plt.ylabel("States")
plt.show()

# plt.plot()      # TODO expectation of receiving the US after the CS
# plt.title("Expectation of receiving the US after the CS")
# plt.xlabel("Trials")
# plt.ylabel("Expectation of receiving the US")
# plt.show()


## Part 3 -----------------------------------------------------------------------------------------

def simple_heuristic(previous_state, similarity, time):
    
    beliefs = np.empty(2)

    # time-related uncertainty:
    # scaling so that for small intervals of one minute t is close to 1
    # and for higher time value t gets smaller
    t = 1/time * 1/1500

    same_state_p = similarity * t
    other_state_p = (1 - similarity) * t    # TODO problem: similarity is always 0 or 1, so this will always yield 0/1 probs at the end of the heuristic

    # if the previous state was state 0 then s0_p will be close to 0
    s0_p = previous_state * same_state_p + (1-previous_state) * other_state_p
    # if the previous state was state 1 then s1_p will be close to 1
    s1_p = previous_state * other_state_p + (1-previous_state) * same_state_p

    beliefs[0] = s0_p / (s0_p + s1_p)
    beliefs[1] = s1_p / (s0_p + s1_p)

    return beliefs


beliefs_over_time = np.empty((101, 2))
# the animal believes to be in state 0 with a probability of 1 before the the first trial
beliefs_over_time[0] = np.array([1,0])

for i in range(1, 101):

    if i == 1:
        similarity = 0.99          # for shortly before the 2nd trial we don't know the second last US
    elif us[i-1] == us[i-2]:
        similarity = 0.99
    else:
        similarity = 0.01

    # TODO use the actual states array here?? if not what then?
    beliefs_over_time[i] = simple_heuristic(states[i-1], similarity, time[i-1])

print(beliefs_over_time[100])

plt.plot(beliefs_over_time[:,0], color="red")
plt.plot(beliefs_over_time[:,1], color="blue")
plt.title("Belief of states over time")
plt.xlabel("Trials")
plt.ylabel("Belief")
plt.show()


## Part 4 -----------------------------------------------------------------------------------------

w_s0 = np.empty(100)            # association weights in state 0
w_s1 = np.empty(100)            # association weights in state 1

w_s0[0], w_s1[0] = 0, 0         # initially weights are zero

e = 0.1                         # learning rate

# TODO this is not quite right yet!
for i in range(1, 100):
    v = w_s0[i-1] * cs[i]
    d = us[i] - v
    w_s0[i] = w_s0[i-1] + e * d * cs[i] * beliefs_over_time[i][0]

    v = w_s1[i-1] * cs[i]
    d = us[i] - v
    w_s1[i] = w_s1[i-1] + e * d * cs[i] * beliefs_over_time[i][1]

plt.plot(w_s0, color="red", label="weights state 0")
plt.plot(w_s1, color="blue", label="weights state 1")
plt.title("State-dependent weights over trials")
plt.xlabel("Trials")
plt.ylabel("Weight")
plt.show()
