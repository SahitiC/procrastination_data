# is evaluating a likelihood actually so intractable?
import task_structure
import mdp_algms
import numpy as np
from scipy.optimize import minimize


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p


def likelihood_basic_model(x,
                           states, actions, horizon, reward_shirk, reward_thr,
                           reward_extra, effort_work, beta,
                           data):
    """
    x = free params of model
    """
    discount_factor = x[0]
    efficacy = x[1]
    # finish defining structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra)

    # total reward= reward+effort
    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    # calculating likelihood
    nllkhd = 0

    for i_time in range(len(data[1:])):

        partial = 0
        # enumerate over all posible actions for the observed state
        for i_a, action in enumerate(actions[data[i_time]]):

            partial += (
                softmax_policy(Q_values[data[i_time]][:, i_time], beta)[action]
                * T[data[i_time]][action][data[i_time+1]])

        nllkhd = nllkhd - np.log(partial)

    return nllkhd


def maximum_likelihood_estimate_basic(states, actions, horizon, reward_shirk,
                                      reward_thr, reward_extra, effort_work,
                                      beta, data):

    nllkhd = np.inf
    # repeat likelihood optimisation for different initial values
    for i in range(5):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        efficacy = np.random.uniform(0, 1)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_basic_model,
                          x0=[discount_factor, efficacy],
                          args=(states, actions, horizon, reward_shirk,
                                reward_thr, reward_extra, effort_work, beta,
                                data),
                          bounds=((0, 1), (0, 1)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_basic_model(
            result.x, states, actions, horizon, reward_shirk, reward_thr,
            reward_extra, effort_work, beta, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result.x
            print(
                f"current estimate for discount factor and efficacy {result.x}"
                f" with neg log likelihood {nllkhd}")

    return final_result
