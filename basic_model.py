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


def get_reward_functions_immediate(states, reward_work, reward_shirk,
                                   effort_work, effort_shirk):
    """
    construct reward function for immediate reward case
    """
    reward_func = []

    # reward for actions (depends on current state and next state)
    reward_func.append([np.array([effort_work, reward_work + effort_work, 0]),
                        np.array([reward_shirk, 0, 0])])
    reward_func.append([np.array([0, effort_work, reward_work + effort_work]),
                        np.array([0, reward_shirk, 0])])
    reward_func.append([np.array([0, 0, reward_shirk])])

    # reward from final evaluation
    reward_func_last = np.array([-1*reward_work, 0, 0])

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
N_INTERMEDIATE_STATES = 1
# intermediate + initial and finished states (2)
STATES = np.arange(2 + N_INTERMEDIATE_STATES)

# actions available in each state
ACTIONS = np.full(len(STATES), np.nan, dtype=object)
# actions for all but final state
ACTIONS[:-1] = [['work', 'shirk']
                for i in range(len(STATES)-1)]
# actions for final state
ACTIONS[-1] = ['completed']

HORIZON = 10  # deadline
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.6  # self-efficacy (probability of progress on working)

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

T = get_transition_prob(STATES, EFFICACY)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    reward_func, reward_func_last, T)
