from scipy.stats import binom
import seaborn as sns
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

HORIZON = 110  # no. of weeks for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.5  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 20000.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

# %%

# self paced (immediate) course rewards
reward_func = []
for state_current in range(len(STATES)):

    reward_temp = np.zeros((len(ACTIONS[state_current]), len(STATES)))

    for action in range(len(ACTIONS[state_current])):

        reward_temp[action, 0:state_current+action+1] = ((len(STATES)-1-action)
                                                         * REWARD_SHIRK)

    if state_current < 14:

        for i, action in enumerate(ACTIONS[state_current]
                                   [14-state_current:22-state_current+1]):

            reward_temp[action, 14:action+state_current+1] += (
                14*REWARD_THR
                + np.arange(0, action+state_current+1-14, step=1)*REWARD_EXTRA)

        for i, action in enumerate(ACTIONS[state_current]
                                   [22-state_current+1:]):

            reward_temp[action, 14:23] += np.arange(14*REWARD_THR,
                                                    16.25*REWARD_THR,
                                                    step=REWARD_THR/4)
            reward_temp[action, 23:action+state_current+1] += 16*REWARD_THR

    elif state_current >= 14 and state_current < 22:

        for i, action in enumerate(ACTIONS[state_current]
                                   [:22-state_current+1]):

            reward_temp[action, state_current+1: action+state_current+1] += (
                np.arange(1, action+1)*REWARD_THR/4)

       # reward_temp[22-state_current+1:, :] = reward_temp[22-state_current, :]

    reward_func.append(reward_temp)

# delayed rewards for final course rewards
reward_func = []
for state_current in range(len(STATES)):

    reward_temp = np.zeros((len(ACTIONS[state_current]), len(STATES)))

    for action in range(len(ACTIONS[state_current])):

        reward_temp[action, 0:state_current+action+1] = ((len(STATES)-1-action)
                                                         * REWARD_SHIRK)

    reward_func.append(reward_temp)

effort_func = []
for state_current in range(len(STATES)):

    effort_temp = np.zeros((len(ACTIONS[state_current]), len(STATES)))

    for i, action in enumerate(ACTIONS[state_current]):

        effort_temp[action, :] = action * EFFORT_WORK

    effort_func.append(effort_temp)

total_reward_func_last = np.zeros(len(STATES))
# np.zeros(len(STATES))
# np.arange(0, STATES_NO, 1)*REWARD_THR
total_reward_func_last[14:22+1] = (14*REWARD_THR
                                   + np.arange(0, 22-14+1)*REWARD_EXTRA)
total_reward_func_last[23:] = 14*REWARD_THR + 8*REWARD_EXTRA

total_reward_func = []
for state_current in range(len(STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

T = []
for state_current in range(len(STATES)):

    T_temp = np.zeros((len(ACTIONS[state_current]), len(STATES)))

    for i, action in enumerate(ACTIONS[state_current]):

        T_temp[action, state_current:state_current+action+1] = (
            binom(action, EFFICACY).pmf(np.arange(action+1))
        )

    T.append(T_temp)

# transition functions with decreasing efficacy
T = []
for i_timestep in range(HORIZON):
    T_t = []
    efficacy = EFFICACY * (1 - (i_timestep / HORIZON))

    for state_current in range(len(STATES)):

        T_temp = np.zeros((len(ACTIONS[state_current]), len(STATES)))

        for i, action in enumerate(ACTIONS[state_current]):

            T_temp[action, state_current:state_current+action+1] = (
                binom(action, efficacy).pmf(np.arange(action+1))
            )

        T_t.append(T_temp)
    T.append(T_t)

# %%

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

efficacy_actual = EFFICACY
T_actual = []
for state_current in range(len(STATES)):

    T_temp = np.zeros((len(ACTIONS[state_current]), len(STATES)))

    for i, action in enumerate(ACTIONS[state_current]):

        T_temp[action, state_current:state_current+action+1] = (
            binom(action, efficacy_actual).pmf(np.arange(action+1))
        )

    T_actual.append(T_temp)

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
