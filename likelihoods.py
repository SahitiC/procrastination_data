# is evaluating a likelihood actually so intractable?
import task_structure
import mdp_algms
import numpy as np


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p


def likelihood_basic_model(data, states, actions, horizon, discount_factor,
                           efficacy, reward_shirk, reward_thr, reward_extra,
                           effort_work, beta):

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
