import numpy as np
import matplotlib.pyplot as plt
import mdp_algms
import task_structure
import likelihoods

# %%
# generate some data
# define params:

# states of markov chain
STATES_NO = 22+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# actions = no. of units to complete in each state
# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 15  # no. of weeks for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.9  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3
BETA = 7

# get task structure

reward_func = task_structure.reward_no_immediate(STATES, ACTIONS, REWARD_SHIRK)

effort_func = task_structure.effort(STATES, ACTIONS, EFFORT_WORK)

total_reward_func_last = task_structure.reward_final(STATES, REWARD_THR,
                                                     REWARD_EXTRA)

total_reward_func = []
for state_current in range(len(STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

T = task_structure.T_binomial(STATES, ACTIONS, EFFICACY)

# get policy

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

# generate data - forward runs

initial_state = 0
s, a = mdp_algms.forward_runs_prob(
    likelihoods.softmax_policy, Q_values, ACTIONS, initial_state,
    HORIZON, STATES, T, BETA)

# %%
# recover params
