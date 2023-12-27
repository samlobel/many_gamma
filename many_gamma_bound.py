# Computes the coefficients. Then gives a maximal error bewteen any two.
# It'll do the second part through numerical integration. Either just
# L1 error, or maybe "most more" and "most less"

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import math

def compute_coefficients(known_gammas: tuple, gamma_target: float, lstsq=False, regularization=0.0):
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
        # A += regularization * np.diag(1 / (1 - np.array(known_gammas))) * (1 / (1 - np.array(known_gammas))).mean()
        ## I did need eye I think. A += regularization # I don't think we need the eye thing here, we want to regularize each term.

    if lstsq:
        coefficients = linalg.lstsq(A, b)[0].tolist()
    else:
        coefficients = linalg.solve(A, b).tolist()
    # print(coefficients)
    return coefficients

def get_constraint_matrix(gammas, lstsq=False, regularization=0.0):
    gammas = np.array(gammas).tolist() # make it a list
    matrix = []
    for i in range(len(gammas)):
        this_gamma = gammas[i]
        other_gammas = gammas[:i] + gammas[i+1:]
        coefficients = compute_coefficients(other_gammas, this_gamma, lstsq=lstsq, regularization=regularization)


        coefficients = coefficients[:i] + [0] + coefficients[i:]
        matrix.append(coefficients)

    matrix = np.array(matrix)
    assert np.max(matrix * np.eye(len(gammas))) == 0, "all should be zero"
    return matrix

def get_constraint_matrix_rectangular(known_gammas, gammas_to_approximate, lstsq=True, regularization=0.0):
    gammas = np.array(known_gammas).tolist() # make it a list
    matrix = []
    for i in range(len(gammas_to_approximate)):
        coefficients = compute_coefficients(known_gammas, gammas_to_approximate[i], lstsq=lstsq, regularization=regularization)
        matrix.append(coefficients)

    matrix = np.array(matrix)
    return matrix


def test_compute_coefficients_same():
    assert np.allclose(compute_coefficients([0.99], 0.99), [1, ]), compute_coefficients([0.99], 0.99)
    assert np.allclose(compute_coefficients([0.9], 0.9), [1, ]), compute_coefficients([0.9], 0.9)

def make_comparison_plot(known_gammas, gamma_target, coefficients=None, write_coefficients=False, save_path=None, title=None):
    if coefficients is None:
        coefficients = compute_coefficients(known_gammas, gamma_target)
    print(coefficients)
    final_x = np.log(0.01) / np.log(gamma_target) # When gamma_target^i = 0.01, after this point not much contribution
    x_axis = np.linspace(0, final_x, 200) # not really enough for what we're looking for
    v_orig = gamma_target ** x_axis
    v_combined = np.zeros_like(x_axis)
    for c, g in zip(coefficients, known_gammas):
        v_combined += c * (g ** x_axis)
    
    pos_diff, neg_diff = get_difference(known_gammas, coefficients, gamma_target)

    # title = fr"-{neg_diff:.4f} < $\Delta$V < {pos_diff:.4f}"
    plt.plot(x_axis, v_orig, label="Original")
    plt.plot(x_axis, v_combined, label="Combined")
    if title:
        plt.title(title)
    plt.legend()

    # caption_string = f"Known gammas: {known_gammas}\nTarget gamma: {gamma_target}"
    gamma_string = ", ".join([f"{g:.3f}" for g in known_gammas])
    caption_string = f"Known gammas: {gamma_string}\nTarget gamma: {gamma_target}\n -{neg_diff:.4f} < Delta V < {pos_diff:.4f}"
    # caption_string = f"Known gammas: {known_gammas}\nTarget gamma: {gamma_target}\n -{neg_diff:.4f} < Delta V < {pos_diff:.4f}"
    plt.subplots_adjust(bottom=0.3)
    plt.annotate(
                caption_string,
                # 'This is a \ncentered text box!', 
                xy=(0.5, -0.2), # (x, y) position with respect to plot
                xycoords='axes fraction', # use axes fraction coordinates system
                textcoords='offset points', # use offset points from the specified xy position
                ha='center', # horizontal alignment is center
                va='center', # vertical alignment is center
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='blue', facecolor='aliceblue'),
                fontsize=12)

    if write_coefficients:
        plt.subplots_adjust(bottom=0.3)
        coefficient_string = ", ".join([f"{c:.2f}" for c in coefficients])
        coefficient_string = f"coefficients: {coefficient_string}"
        plt.annotate(
            coefficient_string,
            # 'This is a \ncentered text box!', 
            xy=(0.5, -0.4), # (x, y) position with respect to plot
            xycoords='axes fraction', # use axes fraction coordinates system
            textcoords='offset points', # use offset points from the specified xy position
            ha='center', # horizontal alignment is center
            va='center', # vertical alignment is center
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='blue', facecolor='aliceblue'),
            fontsize=12)



    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    

def get_difference(known_gammas, coefficients, gamma_target, final_step = 1000):
    # Numerically integrates upper and lower difference bounds.
    # Pos diff is the most that combined could possibly be above target.
    # Neg diff is the most that combined could possibly be below target.
    # everything should be positive
    # print(len(known_gammas))
    # print(len(coefficients))
    assert len(known_gammas) == len(coefficients), f"{len(known_gammas)} != {len(coefficients)}"
    def output(x):
        v_combined = 0
        for c, g in zip(coefficients, known_gammas):
            v_combined += c * (g ** x)
        return v_combined
    x_axis = np.linspace(0, final_step, final_step + 1)
    combined_output = output(x_axis)
    target_output = gamma_target ** x_axis
    diff = combined_output - target_output
    pos_diff = np.maximum(diff, 0).sum()
    neg_diff = -1 * np.minimum(diff, 0).sum()

    return pos_diff, neg_diff

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


def get_upper_and_lower_bounds(gammas, coefficient_matrix, final_step=10000):
    allowed_aboves = []
    allowed_belows = []
    for i, coefficient_list in enumerate(coefficient_matrix):
        gamma = gammas[i]
        allowed_above, allowed_below = get_difference(gammas, coefficient_list, gamma, final_step=final_step)
        allowed_aboves.append(allowed_above)
        allowed_belows.append(allowed_below)

    return np.array(allowed_aboves), np.array(allowed_belows)


def _make_plots(gamma_to_estimate, num_points, regularization):
    print("gamma_to_estimate", gamma_to_estimate, "num_points", num_points, "regularization", regularization)
    points = np.linspace(0.0, 0.9990, num_points)
    coefficients = compute_coefficients(points, gamma_to_estimate, regularization=regularization)
    plt.plot(points, coefficients)
    plt.title(f"Summed coefficients for reg={regularization}: " + str(sum(coefficients)))
    plt.show()

def _get_recreation_for_spacing(num_points=10, regularization=0., start=0., end=0.999, gammas=None):
    # See how well it recreates various things.
    # I should look at a whole range of things. That would be more informative wouldn't it.
    # I'll look at the size of constraints first.
    # uniform_points = np.linspace(0.0, 0.9990, num_points)
    gammas = gammas or get_even_spacing(start, end, num_points)
    # gammas = np.linspace(start, end, num_points)

    # recreation_targets = np.linspace(0.0, 0.999, 1000)
    # recreation_targets = np.linspace(0.9, 0.999, 1000)
    recreation_targets = np.linspace(0.97, 0.997, 100)
    # import ipdb; ipdb.set_trace()
    coefficients_even = [compute_coefficients(gammas, rt, regularization=regularization) for rt in recreation_targets]
    # coefficients_uniform = [compute_coefficients(uniform_points, rt, regularization=regularization) for rt in recreation_targets]
    difference_even = [sum(get_difference(gammas, c, rt)) for c, rt in zip(coefficients_even, recreation_targets)]
    # difference_uniform = [sum(get_difference(uniform_points, c, rt)) for c, rt in zip(coefficients_uniform, recreation_targets)]
    plt.plot(recreation_targets, difference_even, label="Even Spacing")
    # plt.plot(recreation_targets, difference_uniform, label="Uniform Spacing")
    # plt.xscale("log")
    plt.legend()
    plt.show()

    pass

def compare_spacings():
    # For perfect solver, how do different input spacings compare?
    regularization = 1.0
    lower, upper, num = 0.8, 0.997, 100
    linspace_spacing = np.linspace(lower, upper, num)
    even_spacing = np.array(get_even_spacing(lower, upper, num))
    log_spacing = np.array(get_even_log_spacing(lower, upper, num))

    gamma_targets = np.linspace(0.8, 0.997, 1000)

    coefficients_linspace = [compute_coefficients(linspace_spacing, gt, regularization=regularization) for gt in gamma_targets]
    coefficients_even = [compute_coefficients(even_spacing, gt, regularization=regularization) for gt in gamma_targets]
    coefficients_log = [compute_coefficients(log_spacing, gt, regularization=regularization) for gt in gamma_targets]

    difference_linspace = [sum(get_difference(linspace_spacing, c, gt)) for c, gt in zip(coefficients_linspace, gamma_targets)]
    difference_even = [sum(get_difference(even_spacing, c, gt)) for c, gt in zip(coefficients_even, gamma_targets)]
    difference_log = [sum(get_difference(log_spacing, c, gt)) for c, gt in zip(coefficients_log, gamma_targets)]

    plt.plot(gamma_targets, np.array(difference_linspace) * (1 - gamma_targets), label="Linspace")
    plt.plot(gamma_targets, np.array(difference_even) * (1 - gamma_targets), label="Even Spacing")
    plt.plot(gamma_targets, np.array(difference_log) * (1 - gamma_targets), label="Log Spacing")

    # plt.plot(gamma_targets, difference_linspace, label="Linspace")
    # plt.plot(gamma_targets, difference_even, label="Even Spacing")
    # plt.plot(gamma_targets, difference_log, label="Log Spacing")
    plt.legend()
    plt.show()


def _silly():
    # So maybe I want to constrain it so that it always adds up to 1? Seems reasonable.
    # But then it feels like it'll obviously not be a linear equation.
    print(compute_coefficients([0.9], 0.9, regularization=0.1))



if __name__ == '__main__':
    compare_spacings()
    exit()
    # for regularization in [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    # _silly()
    # exit()
    gammas = [0.97, 0.971, 0.982, 0.982, 0.9835, 0.99, 0.991, 0.9915, 0.995, 0.995, 0.996, 0.9968, 0.997]
    _get_recreation_for_spacing(gammas=gammas, regularization=0.1, start = 0.97, end=0.997)
    # _get_recreation_for_spacing(10, regularization=0.1, start = 0.97, end=0.997)
    exit()
    # _make_plots(0.5, 1000, regularization=0)
    # for num_points in [10, 30, 100, 300, 1000, 3000]:
    #     _make_plots(0.5, num_points, regularization=1.0)
    # exit()
    # for regularization_exponent in list(range(-5, 5)):
    #     regularization = 10**regularization_exponent
    #     _make_plots(0.5, 1000, regularization=regularization)
    # exit()
    for i in range(20, 21, 5):
    # for i in range(29, 30):
        for regularization in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-10, 0]:
            print(i, regularization)
            # even_spaced = get_even_spacing(0.9, 0.99, i)
            even_spaced = get_even_spacing(0.1, 0.99, i)
            constraint_matrix_solve = get_constraint_matrix(even_spaced, lstsq=False, regularization=regularization)
            constraint_matrix_lstsq = get_constraint_matrix(even_spaced, lstsq=True, regularization=regularization)
            print(np.array(constraint_matrix_lstsq).sum(axis=0))
            print(np.array(constraint_matrix_lstsq).sum(axis=1))
            continue

            biggest_diff = np.abs(np.array(constraint_matrix_lstsq) - np.array(constraint_matrix_solve)).max()
            upper_bounds_solve, lower_bounds_solve = get_upper_and_lower_bounds(even_spaced, constraint_matrix_solve)
            upper_bounds_lstsq, lower_bounds_lstsq = get_upper_and_lower_bounds(even_spaced, constraint_matrix_lstsq)

            print("biggest diff: ", biggest_diff)
            print("min max solve: ", np.array(constraint_matrix_solve).min(), np.array(constraint_matrix_solve).max())
            print("min max lstsq: ", np.array(constraint_matrix_lstsq).min(), np.array(constraint_matrix_lstsq).max())
            print("upper bounds solve", upper_bounds_solve)
            print("upper bounds lstsq", upper_bounds_lstsq)            
    exit()

    # for i in range(29, 30):
    #     print(i)
    #     even_spaced = get_even_spacing(0.9, 0.99, i)
    #     print(even_spaced)
    #     constraint_matrix_solve = get_constraint_matrix(even_spaced, lstsq=False)
    #     constraint_matrix_lstsq = get_constraint_matrix(even_spaced, lstsq=True)
    #     biggest_diff = np.abs(np.array(constraint_matrix_lstsq) - np.array(constraint_matrix_solve)).max()
    #     upper_bounds_solve, lower_bounds_solve = get_upper_and_lower_bounds(even_spaced, constraint_matrix_solve)
    #     upper_bounds_lstsq, lower_bounds_lstsq = get_upper_and_lower_bounds(even_spaced, constraint_matrix_lstsq)

    #     print("biggest diff: ", biggest_diff)
    #     print("min max solve: ", np.array(constraint_matrix_solve).min(), np.array(constraint_matrix_solve).max())
    #     print("min max lstsq: ", np.array(constraint_matrix_lstsq).min(), np.array(constraint_matrix_lstsq).max())
    #     print("upper bounds solve", upper_bounds_solve)
    #     print("upper bounds lstsq", upper_bounds_lstsq)
    #     print(constraint_matrix_lstsq)
        
    # make_comparison_plot(even_spaced, 0.99, title="Many even spaced 0.99", save_path="./plots/many_gamma/many_even_spaced_0.99.png", write_coefficients=True)
    # even_spaced = get_even_spacing(0.9, 0.99, 8)[:-1]
    # vals = get_even_spacing(0.9, 0.95, 3)[:-1] + get_even_spacing(0.95, 0.975, 3)[1:]
    # print(vals)
    # make_comparison_plot(vals, 0.95, title="Many even spaced 0.99", write_coefficients=True)
    # make_comparison_plot(np.linspace(0.97, 0.99, 10), 0.01, title="Many even spaced 0.99", write_coefficients=True)
    # make_comparison_plot([0.99], 0.01, title="Many even spaced 0.99", write_coefficients=True)
    # make_comparison_plot([0.99], 0.01, coefficients=[0.0], title="Many even spaced 0.99", write_coefficients=True)


    # make_comparison_plot([0.9, 0.975], 0.95, title="Many even spaced 0.99", write_coefficients=True)


    # make_comparison_plot([0.88, 0.92], 0.9, title="Known on Both Sides", save_path="./plots/many_gamma/known_on_both_sides.png", write_coefficients=True)
    # make_comparison_plot(
    #     [0.80, 0.85], 0.9, title="Two Much Smaller", save_path="./plots/many_gamma/two_much_smaller.png",
    #     write_coefficients=True)
    # make_comparison_plot(
    #     [0.80, 0.81, 0.82, 0.83, 0.84, 0.85], 0.9, title="Six All Smaller", save_path="./plots/many_gamma/six_all_smaller.png",
    #     write_coefficients=True)
    # make_comparison_plot(
    #     [0.85, 0.9], 0.8, title="Two Much Bigger", save_path="./plots/many_gamma/two_much_bigger.png",
    #     write_coefficients=True)
    # make_comparison_plot(
    #     [0.86, 0.87, 0.88, 0.89, 0.90], 0.8, title="Six All Bigger", save_path="./plots/many_gamma/six_all_bigger.png",
    #     write_coefficients=True)
    # make_comparison_plot(
    #     [0.86, 0.87, 0.88, 0.89, 0.90], 0.99, title="Approximating 0.99 from way below", save_path="./plots/many_gamma/point_99_from_way_below.png",
    #     write_coefficients=True)
    # make_comparison_plot(
    #     [0.9, 0.91, 0.92, 0.93, 0.94, 0.95], 0.99, title="Approximating 0.99 from below", save_path="./plots/many_gamma/point_99_from_below.png",
    #     write_coefficients=True)
    # make_comparison_plot(
    #     [0.96, 0.97, 0.98, ], 0.99, title="Approximating 0.99 from close below", save_path="./plots/many_gamma/point_99_from_close_below.png",
    #     write_coefficients=True)
    # make_comparison_plot(
    #     [0.98, 0.995], 0.99, title="Approximating 0.99 from both sides", save_path="./plots/many_gamma/point_99_from_both sides.png",
    #     write_coefficients=True)
