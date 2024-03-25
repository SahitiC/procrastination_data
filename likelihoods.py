# is evaluating a likelihood actually so intractable?
import task_structure
import mdp_algms
import numpy as np
from scipy.optimize import minimize


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p


def calculate_likelihood(data, Q_values, beta, T, actions):
    """
    calculate likelihood of data under model given optimal Q_values, beta,
    transitions and actions available
    """
    nllkhd = 0

    for i_trial in range(len(data)):

        for i_time in range(len(data[i_trial][1:])):

            partial = 0
            # enumerate over all posible actions for the observed state
            for i_a, action in enumerate(actions[data[i_trial][i_time]]):

                partial += (
                    softmax_policy(Q_values[data[i_trial][i_time]]
                                   [:, i_time], beta)[action]
                    * T[data[i_trial][i_time]][action][
                        data[i_trial][i_time+1]])

            nllkhd = nllkhd - np.log(partial)

    return nllkhd


def likelihood_basic_model(x,
                           states, actions, horizon,
                           reward_thr, reward_extra, reward_shirk,
                           beta, data):
    """
    x = free params of model
    """
    discount_factor = x[0]
    efficacy = x[1]
    effort_work = x[2]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_efficacy_gap_model(x,
                                  states, actions, horizon,
                                  reward_thr, reward_extra, reward_shirk,
                                  beta, data):
    """
    x = free params of model
    """
    discount_factor = x[0]
    efficacy_assumed = x[1]
    efficacy_actual = x[2]
    effort_work = x[3]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T_assumed = task_structure.T_binomial(states, actions, efficacy_assumed)

    # optimal policy under assumed efficacy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T_assumed)

    T_actual = task_structure.T_binomial(states, actions, efficacy_actual)

    nllkhd = calculate_likelihood(data, Q_values, beta, T_actual, actions)

    return nllkhd


def likelihood_convex_concave_model(x,
                                    states, actions, horizon, efficacy,
                                    reward_thr, reward_extra,
                                    beta, data):
    """
    x = free params of model
    """
    discount_factor = x[0]
    effort_work = x[1]
    exponent = x[2]
    reward_shirk = x[3]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort_convex_concave(states, actions,
                                                       effort_work, exponent)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy under assumed efficacy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def maximum_likelihood_estimate_basic(states, actions, horizon, reward_thr,
                                      reward_extra, reward_shirk, beta, data,
                                      true_params, initial_real=0, verbose=0):
    """
    inputs - fixed parameters, data
    initial_real: whether to include true parameter as an initial point
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_basic_model,
                                x0=true_params,
                                args=(states, actions, horizon,
                                      reward_thr, reward_extra, reward_shirk,
                                      beta, data),
                                bounds=((0, 1), (0, 1), (None, 0)))
        nllkhd = likelihood_basic_model(
            final_result.x, states, actions, horizon, reward_thr, reward_extra,
            reward_shirk, beta, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor, efficacy,"
                  f" effort_work = {final_result.x} "
                  f"with neg log likelihood = {nllkhd}")

    # repeat likelihood-based optimisation for different initial values
    for i in range(10):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        efficacy = np.random.uniform(0, 1)
        # exponential distribution for beta with lambda = 1 or scale = 1
        # following Wilson and Collins 2019:
        # beta = np.random.exponential(2)
        # reward_shirk = np.random.exponential(0.5)
        effort_work = -1 * np.random.exponential(0.5)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_basic_model,
                          x0=[discount_factor, efficacy, effort_work],
                          args=(states, actions, horizon,
                                reward_thr, reward_extra, reward_shirk,
                                beta, data),
                          bounds=((0, 1), (0, 1), (None, 0)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_basic_model(
            result.x, states, actions, horizon, reward_thr, reward_extra,
            reward_shirk, beta, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print(
                    "current estimate for discount_factor, efficacy,"
                    f" effort_work = {final_result.x} "
                    f"with neg log likelihood = {nllkhd}")

    return final_result


def maximum_likelihood_estimate_efficacy_gap(
        states, actions, horizon, reward_thr, reward_extra, reward_shirk, beta,
        data, true_params, initial_real=0, verbose=0):
    """
    inputs - fixed parameters, data
    initial_real: whether to include true parameter as an initial point
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_efficacy_gap_model,
                                x0=true_params,
                                args=(states, actions, horizon,
                                      reward_thr, reward_extra, reward_shirk,
                                      beta, data),
                                bounds=((0, 1), (0, 1), (0, 1), (None, 0)))
        nllkhd = likelihood_efficacy_gap_model(
            final_result.x, states, actions, horizon, reward_thr, reward_extra,
            reward_shirk, beta, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor, efficacy_assumed,"
                  f" efficacy_gap, effort_work = {final_result.x} "
                  f"with neg log likelihood = {nllkhd}")

    # repeat likelihood-based optimisation for different initial values
    for i in range(10):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        efficacy_assumed = np.random.uniform(0, 1)
        efficacy_actual = np.random.uniform(0, 1)
        effort_work = -1 * np.random.exponential(0.5)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_efficacy_gap_model,
                          x0=[discount_factor, efficacy_assumed,
                              efficacy_actual, effort_work],
                          args=(states, actions, horizon,
                                reward_thr, reward_extra, reward_shirk,
                                beta, data),
                          bounds=((0, 1), (0, 1), (0, 1), (None, 0)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_efficacy_gap_model(
            result.x, states, actions, horizon, reward_thr, reward_extra,
            reward_shirk, beta, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print(
                    "current estimate for discount_factor, efficacy_assumed,"
                    f" efficacy_actual, effort_work = {final_result.x} "
                    f"with neg log likelihood = {nllkhd}")

    return final_result


def maximum_likelihood_estimate_convex_concave(
        states, actions, horizon, efficacy, reward_thr, reward_extra,
        beta, data, true_params, initial_real=0, verbose=0):
    """
    inputs - fixed parameters, data
    initial_real: whether to include true parameter as an initial point
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_convex_concave_model,
                                x0=true_params,
                                args=(states, actions, horizon, efficacy,
                                      reward_thr, reward_extra,
                                      beta, data),
                                bounds=((0, 1), (None, 0), (0, None),
                                        (0, None)))
        nllkhd = likelihood_convex_concave_model(
            final_result.x, states, actions, horizon, efficacy, reward_thr,
            reward_extra, beta, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor, effort_work,"
                  f" exponent = {final_result.x} "
                  f"with neg log likelihood = {nllkhd}")

    # repeat likelihood-based optimisation for different initial values
    for i in range(10):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        effort_work = -1 * np.random.exponential(0.5)
        exponent = np.random.gamma(2.5, 0.5)
        reward_shirk = np.random.exponential(0.5)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_convex_concave_model,
                          x0=[discount_factor, effort_work, exponent,
                              reward_shirk],
                          args=(states, actions, horizon, efficacy,
                                reward_thr, reward_extra,
                                beta, data),
                          bounds=((0, 1), (None, 0), (0, None), (0, None)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_convex_concave_model(
            result.x, states, actions, horizon, efficacy, reward_thr,
            reward_extra, beta, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print(
                    "current estimate for discount_factor, effort_work,"
                    f" exponent = {final_result.x} "
                    f"with neg log likelihood = {nllkhd}")

    return final_result
