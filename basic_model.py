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


def get_reward_functions(states, reward_pass, reward_fail, reward_shirk,
                         reward_completed, effort_work):
    """
    construct reward function
    """
    # reward from actions within horizon
    reward_func = []
    # rewards in non-completed states
    reward_func = [([effort_work, reward_shirk])
                   for i in range(len(states)-1)]
    # reward in completed state
    reward_func.append([reward_completed])

    # reward from final evaluation
    reward_func_last = np.linspace(reward_fail, reward_pass, len(states))

    return reward_func, reward_func_last


def get_transition_prob(states, efficacy):
    """
    construct reward function for immediate reward case
    """
    T = []

    # for 3 states:
    T.append([np.array([1-efficacy, efficacy, 0]),
              np.array([1, 0, 0])])  # transitions for work, shirk
    T.append([np.array([0, 1-efficacy, efficacy]),
              np.array([0, 1, 0])])  # transitions for work, shirk
    T.append([np.array([0, 0, 1])])  # transitions for completed

#    # for 2 states:
#    T[0] = [ np.array([1-efficacy, efficacy]),
#             np.array([1, 0]) ] # transitions for work, shirk
#    T[1] = [ np.array([0, 1]) ] # transitions for completed

    return T


def deterministic_policy(a):
    p = np.where(a == np.max(a), 1, 0)
    return p / sum(p)


def softmax_policy(a, beta):
    return a

# %%
# instantiate MDP


# states of markov chain
STATES_NO = 22+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# actions = no. of units to complete in each state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 110  # no. of days for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.3  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 7.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.01
EFFORT_WORK = -0.2

# %%

# immediate rewards for final course rewards
reward_func = []
for state_current in range(len(STATES)):

    reward_temp = np.zeros((len(ACTIONS[state_current]), len(STATES)))

    for action in range(len(ACTIONS[state_current])):

        reward_temp[action, 0:state_current+action+1] = ((len(STATES)-1-action)
                                                         * REWARD_SHIRK)

    reward_func.append(reward_temp)

# self paced course rewards
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

        for i, action in enumerate(ACTIONS[state_current][22-state_current:]):

            reward_temp[action, 14:23] += np.arange(14*REWARD_THR,
                                                    16.25*REWARD_THR,
                                                    step=REWARD_THR/4)
            reward_temp[action, 23:action+state_current+1] += 16*REWARD_THR

    elif state_current >= 14 and state_current < 22:

        for i, action in enumerate(ACTIONS[state_current]
                                   [:22-state_current+1]):

            reward_temp[action, state_current+1: action+state_current+1] += (
                np.arange(1, action+1)*REWARD_THR/4)

        reward_temp[22-state_current+1:, :] = reward_temp[22-state_current, :]

    reward_func.append(reward_temp)

effort_func = []
for state_current in range(len(STATES)):

    effort_temp = np.zeros((len(ACTIONS[state_current]), len(STATES)))

    for i, action in enumerate(ACTIONS[state_current]):

        effort_temp[action, :] = action * EFFORT_WORK

    effort_func.append(effort_temp)


T = []
for state_current in range(len(STATES)):

    T_temp = np.zeros((len(ACTIONS[state_current]), len(STATES)))

    for i, action in enumerate(ACTIONS[state_current]):

        T_temp[action, state_current:state_current+action+1] = (
            binom(action, EFFICACY).pmf(np.arange(action+1))
        )

    T.append(T_temp)

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

# %%

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

initial_state = 0
s, a, v = mdp_algms.forward_runs(
    policy_opt, V_opt, initial_state, HORIZON, STATES, T)

plt.plot(s)

# %%

V_opt = np.full((len(STATES), HORIZON+1), np.nan)
policy_opt = np.full((len(STATES), HORIZON), np.nan)
Q_values = np.full(len(STATES), np.nan, dtype=object)

for i_state, state in enumerate(STATES):

    # V_opt for last time-step
    V_opt[i_state, -1] = total_reward_func_last[i_state]
    # arrays to store Q-values for each action in each state
    Q_values[i_state] = np.full((len(ACTIONS[i_state]), HORIZON), np.nan)

# backward induction to derive optimal policy
for i_timestep in range(HORIZON-1, -1, -1):

    for i_state, state in enumerate(STATES):

        Q = np.full(len(ACTIONS[i_state]), np.nan)

        for i_action, action in enumerate(ACTIONS[i_state]):

            # q-value for each action (bellman equation)
            Q[i_action] = (T[i_state][i_action]
                           @ total_reward_func[i_state][i_action].T
                           + DISCOUNT_FACTOR * (T[i_state][i_action]
                                                @ V_opt[STATES,
                                                        i_timestep+1]))

        # find optimal action (which gives max q-value)
        V_opt[i_state, i_timestep] = np.max(Q)
        policy_opt[i_state, i_timestep] = np.argmax(Q)
        Q_values[i_state][:, i_timestep] = Q
