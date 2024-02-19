import numpy as np
import matplotlib.pyplot as plt
import mdp_algms
import task_structure
import likelihoods
import time

# %%


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p

# %%
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

# %%
# generate some data (basic model)


def gen_data_basic(discount_factor, efficacy):

    # get task structure
    reward_func = task_structure.reward_no_immediate(
        STATES, ACTIONS, REWARD_SHIRK)

    effort_func = task_structure.effort(STATES, ACTIONS, EFFORT_WORK)

    total_reward_func_last = task_structure.reward_final(STATES, REWARD_THR,
                                                         REWARD_EXTRA)

    total_reward_func = []
    for state_current in range(len(STATES)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(STATES, ACTIONS, efficacy)

    # get policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        STATES, ACTIONS, HORIZON, discount_factor,
        total_reward_func, total_reward_func_last, T)

    # generate data - forward runs
    initial_state = 0
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, ACTIONS, initial_state,
        HORIZON, STATES, T, BETA)

    return s


# %%
# example recovery

# generate data
data = gen_data_basic(DISCOUNT_FACTOR, EFFICACY)
# recover params given data (=s)
start = time.time()
mle_result = likelihoods.maximum_likelihood_estimate_basic(
    STATES, ACTIONS, HORIZON, REWARD_SHIRK, REWARD_THR, REWARD_EXTRA,
    EFFORT_WORK, BETA, data)
end = time.time()
print(end-start)

# systematic recovery for many params
start = time.time()
params = []
for i in range(200):
    # generate random parameters
    discount_factor = np.random.uniform(0, 1)
    efficacy = np.random.uniform(0, 1)
    # generate data
    data = gen_data_basic(discount_factor, efficacy)
    # recover params given data
    mle_result = likelihoods.maximum_likelihood_estimate_basic(
        STATES, ACTIONS, HORIZON, REWARD_SHIRK, REWARD_THR, REWARD_EXTRA,
        EFFORT_WORK, BETA, data)
    params.append([discount_factor, efficacy,
                   mle_result[0], mle_result[1]])
end = time.time()
print(end-start)

# %%
params = np.array(params)
colors = np.array(['tab:blue', 'tab:orange'])
efficacys_color = np.where(params[:, 1] < 0.35, 1, 0)
plt.figure()
plt.scatter(params[:, 0], params[:, 2],
            color=colors[efficacys_color])
plt.xlabel('true discount factor')
plt.ylabel('estimated discount factor')
plt.legend()
plt.figure()
discounts_color = np.where(params[:, 0] < 0.2, 1, 0)
plt.scatter(params[:, 1], params[:, 3], color=colors[discounts_color])
plt.xlabel('true efficacy')
plt.ylabel('estimated efficacy')
