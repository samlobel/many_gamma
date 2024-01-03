import numpy as np
import torch
from torch import nn
from scipy import linalg

def get_difference(known_gammas, coefficients, gamma_target, final_step = 1000):
    # Numerically integrates upper and lower difference bounds.
    # Pos diff is the most that combined could possibly be above target.
    # Neg diff is the most that combined could possibly be below target.
    # I don't need symmetric here because I could just add them up later.
    def output(x):
        v_combined = 0
        for c, g in zip(coefficients, known_gammas):
            v_combined += c * (g ** x)
        return v_combined
    x_axis = np.linspace(0, final_step, final_step + 1)
    combined_output = output(x_axis)
    target_output = gamma_target ** x_axis
    diff = combined_output - target_output
    allowed_above = np.maximum(diff, 0).sum()
    allowed_below = -1 * np.minimum(diff, 0).sum()
    assert allowed_below >= 0
    assert allowed_above >= 0
    return allowed_above, allowed_below


def compute_coefficients(known_gammas: tuple, gamma_target: float, regularization=0.0):
    if len(known_gammas) == 0:
        return []
    A = []
    b = []
    for gamma_i in known_gammas:
        A_i = [1 / (1 - gamma_i * gamma_j) for gamma_j in known_gammas]
        b_i = 1 / (1 - gamma_i * gamma_target)
        A.append(A_i)
        b.append(b_i)

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if regularization:
        A += regularization * np.eye(len(known_gammas))
    coefficients = linalg.lstsq(A, b)[0].tolist()
    return coefficients

def get_constraint_matrix(gammas, regularization=0.0, skip_self_map=True, only_from_lower=False):
    # Note that you need to transpose this in order to have matmul(output, coefficients) give the right thing.
    # Maybe we want to fix that here? Not sure. It works when you look at it the other way, coefficients * output. 
    # Not that we ever really need to do that. I should make the change. But in its own commit.
    gammas = np.array(gammas).tolist() # make it a list
    matrix = []
    for i in range(len(gammas)):
        this_gamma = gammas[i]

        # I don't know maybe this is uglier but at least its uglier for less lines.
        if skip_self_map and only_from_lower: other_gammas = gammas[:i]
        elif skip_self_map and not only_from_lower: other_gammas = gammas[:i] + gammas[i+1:]
        elif not skip_self_map and only_from_lower: other_gammas = gammas[:i+1]
        else: other_gammas = gammas

        coefficients = compute_coefficients(other_gammas, this_gamma, regularization=regularization)
        if skip_self_map and only_from_lower: coefficients = coefficients + [0]*(len(gammas) - len(coefficients))
        elif skip_self_map and not only_from_lower: coefficients = coefficients[:i] + [0] + coefficients[i:]
        elif not skip_self_map and only_from_lower: coefficients = coefficients + [0]*(len(gammas) - len(coefficients)) # I could fold this into the first one.
        else: coefficients = coefficients
        assert len(coefficients) == len(gammas)

        matrix.append(coefficients)

    matrix = np.array(matrix)

    if skip_self_map:
        assert np.max(matrix * np.eye(len(gammas))) == 0, "all diags should be zero" # There's also a np.diag I think
    if only_from_lower:
        # assert np.max(np.tril(matrix, -1)) == 0, "all upper triags should be zero"
        assert np.max(np.triu(matrix, 1)) == 0, matrix
    assert matrix.shape == (len(gammas), len(gammas))
    return matrix

def get_upper_and_lower_bounds(gammas, coefficient_matrix, final_step=10000, r_min=-1.0, r_max=1.0):
    # TODO: This assumes it each element is a list of coefficients. Which means that after we transpose it its no longer correct. 
    # Meaning the matmul and this have to work differently. Sad.
    # allowed_aboves is how much above the computed values can be than the actual and still be consistent. 
    # Pretty sure this is right.
    allowed_aboves = []
    allowed_belows = []
    for i, coefficient_list in enumerate(coefficient_matrix):
        gamma = gammas[i]
        allowed_above, allowed_below = get_difference(gammas, coefficient_list, gamma, final_step=final_step)
        allowed_aboves.append(allowed_above)
        allowed_belows.append(allowed_below)
    allowed_aboves, allowed_belows = np.array(allowed_aboves), np.array(allowed_belows)
    weighted_allowed_aboves = r_max*allowed_aboves - r_min*allowed_belows
    weighted_allowed_belows = r_max*allowed_belows - r_min*allowed_aboves

    return weighted_allowed_aboves, weighted_allowed_belows
    # return np.array(allowed_aboves), np.array(allowed_belows)

def get_even_spacing(gamma_small, gamma_big, total_points):
    assert gamma_small < gamma_big
    assert total_points >= 2
    vmax_small = 1 / (1 - gamma_small)
    vmax_big = 1 / (1 - gamma_big)
    spaces = np.linspace(vmax_small, vmax_big, total_points)
    inverted = 1 - 1 / spaces
    return inverted.tolist()

def get_even_log_spacing(gamma_small, gamma_big, total_points):
    # Feels sort of natural, even spacing in log space of 1 - gamma.
    # Lets you do 0 but not 1, which is right.
    assert gamma_small < gamma_big
    assert total_points >= 2
    log_big = np.log(1 - gamma_big)
    log_small = np.log(1 - gamma_small)
    spaces = np.linspace(log_small, log_big, total_points)
    inverted = 1 - np.exp(spaces)
    return inverted.tolist()

if __name__ == '__main__':
    # print(get_even_log_spacing(0.8, 0.99, 10))
    spacing = get_even_log_spacing(0.8, 0.99, 10)
    yes_skip_yes_lower = get_constraint_matrix(spacing, regularization=0.0, skip_self_map=True, only_from_lower=True)
    yes_skip_no_lower = get_constraint_matrix(spacing, regularization=0.0, skip_self_map=True, only_from_lower=False)
    no_skip_yes_lower = get_constraint_matrix(spacing, regularization=0.0, skip_self_map=False, only_from_lower=True)
    no_skip_no_lower = get_constraint_matrix(spacing, regularization=0.0, skip_self_map=False, only_from_lower=False)
    upper_bounds, lower_bounds = get_upper_and_lower_bounds(spacing, yes_skip_yes_lower)

