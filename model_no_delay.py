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
STATES_NO = 22+1  # 22 one extra state for completing nothing
STATES = np.arange(STATES_NO)

# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 15  # no. of weeks for task
DISCOUNT_FACTOR = 0.5  # discounting factor
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 7.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/8  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

# %%
# define environment and reward structure
# rewards as soon as 14 credits are hit

reward_func = task_structure.reward_threshold(
    STATES, ACTIONS, REWARD_SHIRK, REWARD_THR, REWARD_EXTRA)

effort_func = task_structure.effort(STATES, ACTIONS, EFFORT_WORK)

total_reward_func_last = np.zeros(len(STATES))

# total reward= reward+effort
total_reward_func = []
for state_current in range(len(STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

T = task_structure.T_binomial(STATES, ACTIONS, EFFICACY)

# what is the policy for this reward schedule with single discount factor
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

plt.figure()
plt.imshow(policy_opt)
plt.xlabel('time')
plt.ylabel('states')
cbar = plt.colorbar()
cbar.ax.set_ylabel('actions', rotation=270, labelpad=15)

initial_state = 0
beta = 5
plt.figure()
for i in range(20):
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
        T, beta)
    plt.plot(s, color='gray')
plt.plot(s, color='gray', label='softmax noise')

initial_state = 0
s, a, v = mdp_algms.forward_runs(
    policy_opt, V_opt, initial_state, HORIZON, STATES, T)
plt.plot(s, label='deterministic')

plt.xlabel('weeks')
plt.ylabel('units completed')
plt.legend(fontsize=10)

# %%
# what if i limit the number of units possible in each day

# set maximum no. of units that can be finished in a day
MAX_UNITS = 6
ACTIONS_LIM = []
for state_current in range(STATES_NO):

    if state_current + MAX_UNITS <= STATES_NO-1:
        units = MAX_UNITS
    else:
        units = STATES_NO-1-state_current

    ACTIONS_LIM.append(np.arange(units+1))

reward_func = task_structure.reward_threshold(
    STATES, ACTIONS_LIM, REWARD_SHIRK, REWARD_THR, REWARD_EXTRA)

effort_func = task_structure.effort(STATES, ACTIONS_LIM, EFFORT_WORK)

total_reward_func_last = np.zeros(len(STATES))

# total reward = reward+effort
total_reward_func = []
for state_current in range(len(STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

T = task_structure.T_binomial(STATES, ACTIONS_LIM, EFFICACY)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS_LIM, HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

plt.figure()
plt.imshow(policy_opt)
plt.xlabel('time')
plt.ylabel('states')
cbar = plt.colorbar()
cbar.ax.set_ylabel('actions', rotation=270, labelpad=15)

initial_state = 0
beta = 5
plt.figure()
for i in range(20):
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, ACTIONS_LIM, initial_state, HORIZON, STATES,
        T, beta)
    plt.plot(s, color='gray')
plt.plot(s, color='gray', label='softmax noise')

initial_state = 0
s, a, v = mdp_algms.forward_runs(
    policy_opt, V_opt, initial_state, HORIZON, STATES, T)
plt.plot(s, label='deterministic')

plt.xlabel('weeks')
plt.ylabel('units completed')
plt.legend(fontsize=10)

# %%
# what if there is a cost related to the number of units

EXPONENT = 2.3  # to make effort function more convex

reward_func = task_structure.reward_threshold(
    STATES, ACTIONS, REWARD_SHIRK, REWARD_THR, REWARD_EXTRA)

effort_func = task_structure.effort_convex_concave(STATES, ACTIONS,
                                                   EFFORT_WORK, EXPONENT)

total_reward_func_last = np.zeros(len(STATES))

# total reward = reward+effort
total_reward_func = []
for state_current in range(len(STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

T = task_structure.T_binomial(STATES, ACTIONS, EFFICACY)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

plt.figure()
plt.imshow(policy_opt)
plt.xlabel('time')
plt.ylabel('states')
cbar = plt.colorbar()
cbar.ax.set_ylabel('actions', rotation=270, labelpad=15)

initial_state = 0
beta = 7
plt.figure()
for i in range(20):
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
        T, beta)
    plt.plot(s, color='gray')
plt.plot(s, color='gray', label='softmax noise')

initial_state = 0
s, a, v = mdp_algms.forward_runs(
    policy_opt, V_opt, initial_state, HORIZON, STATES, T)
plt.plot(s, label='deterministic')

plt.xlabel('weeks')
plt.ylabel('units completed')
plt.legend(fontsize=10)
