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
# set parameters


# states of markov chain
STATES_NO = 22+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 16  # no. of weeks for task
DISCOUNT_FACTOR_REWARD = 0.5  # discounting factor
DISCOUNT_FACTOR_COST = 0.9
EFFICACY = 0.5  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 1.4  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

# %%
# define environment and reward structure

reward_func = task_structure.reward_threshold(
    STATES, ACTIONS, REWARD_SHIRK, REWARD_THR, REWARD_EXTRA)

effort_func = task_structure.effort(STATES, ACTIONS, EFFORT_WORK)

reward_func_last = np.zeros(len(STATES))
effort_func_last = np.zeros(len(STATES))
T = task_structure.T_binomial(STATES, ACTIONS, EFFICACY)

# %%
# solve for policy given task structure

V_opt_full, policy_opt_full, Q_values_full = (
    mdp_algms.find_optimal_policy_diff_discount_factors(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD,
        DISCOUNT_FACTOR_COST, reward_func, effort_func, reward_func_last,
        effort_func_last, T)
)

# effective policy followed by agent
effective_policy = np.array(
    [[policy_opt_full[HORIZON-1-i][i_s][i] for i in range(HORIZON)]
     for i_s in range(len(STATES))]
)
effective_Q = []
for i_s in range(len(STATES)):
    Q_s_temp = []
    for i in range(HORIZON):
        Q_s_temp.append(Q_values_full[HORIZON-1-i][i_s][:, i])
    effective_Q.append(np.array(Q_s_temp).T)

efficacy_actual = 0.9
T_actual = task_structure.T_binomial(STATES, ACTIONS, efficacy_actual)

initial_state = 0
beta = 5
for i in range(20):
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, effective_Q, ACTIONS, initial_state, HORIZON, STATES,
        T_actual, beta)
    plt.plot(s, color='gray')
plt.plot(s, color='gray', label='with softmax noise')

initial_state = 0
s, a, v = mdp_algms.forward_runs(
    effective_policy, V_opt_full[0], initial_state, HORIZON, STATES, T_actual)

plt.plot(s, label='deterministic')
