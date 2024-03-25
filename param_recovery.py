import numpy as np
import likelihoods
import gen_data
import concurrent.futures


# function to recover free parameters in basic model
def recover_params_basic(input_params):
    """
    inputs: parameters to recover
    """

    discount_factor = input_params[0]
    efficacy = input_params[1]
    effort_work = input_params[2]

    # generate data
    data = []
    for i_trials in range(N_TRIALS):

        s = gen_data.gen_data_basic(
            STATES, ACTIONS, HORIZON, discount_factor, efficacy, BETA,
            REWARD_SHIRK, effort_work, REWARD_THR, REWARD_EXTRA)
        data.append(s)

    # recover params given data
    mle_result = likelihoods.maximum_likelihood_estimate_basic(
        STATES, ACTIONS, HORIZON, REWARD_THR, REWARD_EXTRA, REWARD_SHIRK,
        BETA, data, input_params, initial_real=1)
    params = [
        discount_factor, efficacy, effort_work,
        mle_result.x[0], mle_result.x[1], mle_result.x[2]]
    inv_hessians = mle_result.hess_inv.todense()

    return [params, inv_hessians]


# function to recover free parameters in model with efficacy gap
def recover_params_efficacy_gap(input_params):
    """
    inputs: parameters to recover
    """

    discount_factor = input_params[0]
    efficacy_assumed = input_params[1]
    efficacy_actual = input_params[2]
    effort_work = input_params[3]

    # generate data
    data = []
    for i_trials in range(N_TRIALS):

        s = gen_data.gen_data_efficacy_gap(
            STATES, ACTIONS, HORIZON, discount_factor, efficacy_assumed,
            efficacy_actual, BETA, REWARD_SHIRK, effort_work, REWARD_THR,
            REWARD_EXTRA)
        data.append(s)

    # recover params given data
    mle_result = likelihoods.maximum_likelihood_estimate_efficacy_gap(
        STATES, ACTIONS, HORIZON, REWARD_THR, REWARD_EXTRA, REWARD_SHIRK,
        BETA, data, input_params, initial_real=1)
    params = [
        discount_factor, efficacy_assumed, efficacy_actual, effort_work,
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
DISCOUNT_FACTOR = 0.8  # discounting factor
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0
EFFORT_WORK = -0.2
BETA = 5

N_TRIALS = 1  # no. of trajectories per dataeset for recovery
N = 1  # no of params sets to recover

# %%
# generate iterable list of input params
input_lst = []
for i in range(N):
    # generate random parameters
    discount_factor = np.random.uniform(0.2, 1)
    efficacy_assumed = np.random.uniform(0.35, 1)
    efficacy_actual = np.random.uniform(0.35, 1)
    effort_work = -1 * np.random.exponential(0.5)

    input_lst.append([discount_factor, efficacy_assumed,
                      efficacy_actual, effort_work])

# parallelise code
if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_lst = executor.map(recover_params_efficacy_gap, input_lst)

    result_lst = [*result_lst]
    result = np.array(result_lst, dtype=object)
    # np.save('result.npy', result)
