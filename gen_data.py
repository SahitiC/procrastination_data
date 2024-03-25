"""
module to generate data (trajectories) given for each type of model given
input parameters and task structure
"""

import mdp_algms
import task_structure
import numpy as np


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

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra)

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
    s, a = mdp_algms.forward_runs_prob(softmax_policy, Q_values, actions,
                                       initial_state, horizon, states, T, beta)

    return s


# function to generate a trajctory given parameters using efficacy-gap model
def gen_data_efficacy_gap(states, actions, horizon, discount_factor,
                          efficacy_assumed, efficacy_actual, beta, reward_shirk,
                          effort_work, reward_thr, reward_extra):

    # get reward function
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    # get transitions based on assumed efficacy
    T_assumed = task_structure.T_binomial(states, actions, efficacy_assumed)

    # get policy according to assumed transitions
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T_assumed)

    # get transition prob based on actual efficacy
    T_actual = task_structure.T_binomial(states, actions, efficacy_actual)

    # generate data - forward runs based on actual tranistion prob
    initial_state = 0
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, actions, initial_state, horizon, states,
        T_actual, beta)

    return s
