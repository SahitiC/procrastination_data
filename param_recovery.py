import numpy as np
import mdp_algms
import task_structure
import likelihoods
import concurrent.futures


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p


# function to generate a trajctory given parameters using basic model
def gen_data_basic(states, actions, horizon, discount_factor, efficacy, beta,
                   reward_shirk, effort_work, reward_thr, reward_extra):

    # get reward function
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(states, reward_thr,
                                                         reward_extra)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    # get tranistions
    T = task_structure.T_binomial(states, actions, efficacy)

    # get policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    # generate data - forward runs
    initial_state = 0
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, actions, initial_state,
        horizon, states, T, beta)

    return s


# function to recover free parameters
def recover_params_basic(input_params):
    """
    inputs: parameters to recover
    """

    discount_factor = input_params[0]
    efficacy = input_params[1]
    reward_shirk = input_params[2]
    effort_work = input_params[3]

    # generate data
    data = []
    for i_trials in range(N_TRIALS):

        s = gen_data_basic(
            STATES, ACTIONS, HORIZON, discount_factor, efficacy, BETA,
            reward_shirk, effort_work, REWARD_THR, REWARD_EXTRA)
        data.append(s)

    # recover params given data
    mle_result = likelihoods.maximum_likelihood_estimate_basic(
        STATES, ACTIONS, HORIZON, REWARD_THR, REWARD_EXTRA, BETA, data,
        input_params, initial_real=1)
    params = [
        discount_factor, efficacy, reward_shirk, effort_work,
        mle_result.x[0], mle_result.x[1], mle_result.x[2], mle_result.x[3]]
    inv_hessians = mle_result.hess_inv.todense()

    return [params, inv_hessians]


# %%

# define some standard params:

# states of markov chain
STATES_NO = 22+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# actions = no. of units to complete in each state
# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 15  # no. of weeks for task
DISCOUNT_FACTOR = 0.7  # discounting factor
EFFICACY = 0.6  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3
BETA = 5

N_TRIALS = 3  # no. of trajectories per data
N = 1000  # no of params sets to recover
# generate iterable list of input params
inputs_lst = []
for i in range(N):
    # generate random parameters
    discount_factor = np.random.uniform(0.2, 1)
    efficacy = np.random.uniform(0.35, 1)
    #beta = np.random.exponential(2)
    reward_shirk = np.random.exponential(0.5)
    effort_work = -1 * np.random.exponential(0.5)

    inputs_lst.append([discount_factor, efficacy, reward_shirk, effort_work])

# parallelise code
if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_lst = executor.map(recover_params_basic, inputs_lst)

    result_lst = [*result_lst]
    result = np.array(result_lst, dtype=object)
    np.save('result.npy', result)
