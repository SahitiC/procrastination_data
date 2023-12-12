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

# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 16  # no. of weeks for task
DISCOUNT_FACTOR_REWARD = 0.9  # discounting factor
DISCOUNT_FACTOR_COST = 0.8
EFFICACY = 0.4  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

# %%
reward_func = task_structure.reward_immediate_threshold(
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

# actual policy followed by agent
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

initial_state = 0
s, a, v = mdp_algms.forward_runs(
    effective_policy, V_opt_full[0], initial_state, HORIZON, STATES, T)

plt.plot(s, label='deterministic')

initial_state = 0
beta = 5
for i in range(20):
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, effective_Q, ACTIONS, initial_state, HORIZON, STATES,
        T, beta)
    plt.plot(s, color='gray')
plt.plot(s, color='gray', label='with softmax noise')

# %%

efficacies = np.array([0.2, 0.4, 0.6, 0.8])

for i_e, efficacy in enumerate(efficacies):

    T = task_structure.T_binomial(STATES, ACTIONS, efficacy)

    V_opt_full, policy_opt_full, Q_values_full = (
        mdp_algms.find_optimal_policy_diff_discount_factors(
            STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD,
            DISCOUNT_FACTOR_COST, reward_func, effort_func, reward_func_last,
            effort_func_last, T)
    )

    # actual policy followed by agent
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

    initial_state = 0
    s, a, v = mdp_algms.forward_runs(
        effective_policy, V_opt_full[0], initial_state, HORIZON, STATES, T)

    plt.plot(s, label=f'{efficacy}')

plt.legend()
