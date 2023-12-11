import seaborn as sns
import mdp_algms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2

# %%


def get_reward_functions(states, reward_pass, reward_fail, reward_shirk,
                         reward_completed, effort_work, effort_shirk):
    """
    construct reward function
    """
    # reward from actions within horizon
    reward_func = []
    # rewards in non-completed states
    reward_func = [([effort_work, reward_shirk + effort_shirk])
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

    for i in range(len(states)-1):
        temp1 = np.zeros(len(states))
        temp2 = np.zeros(len(states))
        temp1[i] = 1-efficacy
        temp1[i+1] = efficacy
        temp2[i] = 1
        T.append([temp1, temp2])

    temp1 = np.zeros(len(states))
    temp1[i+1] = 1
    T.append([temp1])

    return T


def get_transition_prob_decreasing(states, efficacy):
    """
    construct reward function for immediate reward case
    """
    T = []

    for i in range(len(states)-1):
        temp1 = np.zeros(len(states))
        temp2 = np.zeros(len(states))
        temp1[i] = 1-efficacy
        temp1[i+1] = efficacy
        temp2[i] = 1
        T.append([temp1, temp2])

    temp1 = np.zeros(len(states))
    temp1[i+1] = 1
    T.append([temp1])

    return T


def deterministic_policy(a):
    p = np.where(a == np.max(a), 1, 0)
    return p / sum(p)


def softmax_policy(a, beta):

    return np.exp(beta*a) / sum(np.exp(beta*a))

# %%
# instantiate MDP


# states of markov chain = units of tasks
STATE_NO = 32+1  # state where nothing is done as an additional state
STATES = np.arange(STATE_NO)

# action = decision to complete x (<remaining units) units
ACTIONS = []
ACTIONS = [np.arange(STATE_NO-i)
           for i in range(len(STATES))]

HORIZON = 110  # no. of days for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.8  # efficacy (probability of progress on working)

# utilities :
REWARD_PASS = 4.0
REWARD_FAIL = -4.0
REWARD_SHIRK = 0.5
EFFORT_WORK = -0.4
EFFORT_SHIRK = -0
REWARD_COMPLETED = REWARD_SHIRK

# %%
reward_func, reward_func_last = get_reward_functions(
    STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK,
    REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)

assumed_efficacy = 0.7

T_assumed = get_transition_prob(STATES, assumed_efficacy)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    reward_func, reward_func_last, T_assumed)

# plots of policies and values
plt.figure(figsize=(8, 6))
for i_state, state in enumerate(STATES):

    # plt.plot(V_opt[i_state], label=f'V*{i_state}',
    # marker=i_state+4, linestyle='--')
    # plt.plot(policy_opt[i_state], label = 'policy*')

    for i_action, action in enumerate(ACTIONS[i_state]):

        plt.plot(Q_values[i_state][i_action, :], label=r'Q' +
                 action, marker=i_state+4, linestyle='--')

    plt.legend()

# %%

BETA = 10
initial_state = 0
d = np.zeros(HORIZON)

T = get_transition_prob(STATES, EFFICACY)

for i in range(10000):
    s, a = mdp_algms.forward_runs(softmax_policy, Q_values, ACTIONS,
                                  initial_state, HORIZON, STATES, T, BETA)
    delta_progress = np.zeros(HORIZON)
    delta_progress[np.where(s[:-1] < s[1:])[0]] = 1
    d = d + delta_progress

plt.plot(d/10000)
