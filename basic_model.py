
import mdp_algms
import task_structure
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2

# %%


def deterministic_policy(a):
    p = np.where(a == np.max(a), 1, 0)
    return p / sum(p)


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p


# %%
# instantiate MDP

# states of markov chain
STATES_NO = 22+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# actions = no. of units to complete in each state
# set maximum no. of units that can be finished in a day
MAX_UNITS = 8
ACTIONS = []
for state_current in range(STATES_NO):

    if state_current + MAX_UNITS <= STATES_NO-1:
        units = MAX_UNITS
    else:
        units = STATES_NO-1-state_current

    ACTIONS.append(np.arange(units+1))

# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 16  # no. of weeks for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.9  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

# %%

reward_func = task_structure.reward_no_immediate(STATES, ACTIONS, REWARD_SHIRK)

effort_func = task_structure.effort(STATES, ACTIONS, EFFORT_WORK)

total_reward_func_last = task_structure.reward_final(STATES, REWARD_THR,
                                                     REWARD_EXTRA)

# total reward= reward+effort
total_reward_func = []
for state_current in range(len(STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

# %%

T = task_structure.T_binomial(STATES, ACTIONS, EFFICACY)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

efficacy_actual = EFFICACY
T_actual = task_structure.T_binomial(STATES, ACTIONS, efficacy_actual)

initial_state = 0
s, a, v = mdp_algms.forward_runs(
    policy_opt, V_opt, initial_state, HORIZON, STATES, T_actual)

plt.plot(s, label='deterministic')

initial_state = 0
beta = 5
for i in range(20):
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
        T_actual, beta)
    plt.plot(s, color='gray')
plt.plot(s, color='gray', label='with softmax noise')

plt.legend(fontsize=10)
plt.xlabel('weeks')
plt.ylabel('units completed')

# %%

T = task_structure.T_binomial_decreasing(STATES, ACTIONS, HORIZON, EFFICACY)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_T_time_dep(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

initial_state = 0
s, a = mdp_algms.forward_runs_T_time_dep(
    deterministic_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
    T)
plt.plot(s, label='deterministic')

initial_state = 0
beta = 5
for i in range(20):
    s, a = mdp_algms.forward_runs_T_time_dep(
        softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
        T, beta)
    plt.plot(s, color='gray')
plt.plot(s, color='gray', label='with softmax noise')

plt.legend(fontsize=10)
