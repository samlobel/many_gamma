# Computes the coefficients. Then gives a maximal error bewteen any two.
# It'll do the second part through numerical integration. Either just
# L1 error, or maybe "most more" and "most less"

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def compute_coefficients(known_gammas: tuple, gamma_target: float):
    A = []
    b = []
    for gamma_i in known_gammas:
        A_i = [1 / (1 - gamma_i * gamma_j) for gamma_j in known_gammas]
        b_i = 1 / (1 - gamma_i * gamma_target)
        A.append(A_i)
        b.append(b_i)

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    coefficients = linalg.solve(A, b).tolist()
    print(coefficients)
    return coefficients

def get_constraint_matrix(gammas):
    gammas = np.array(gammas).tolist() # make it a list
    matrix = []
    for i in range(len(gammas)):
        this_gamma = gammas[i]
        other_gammas = gammas[:i] + gammas[i+1:]
        coefficients = compute_coefficients(other_gammas, this_gamma)


        coefficients = coefficients[:i] + [0] + coefficients[i:]
        matrix.append(coefficients)

    matrix = np.array(matrix)
    assert np.max(matrix * np.eye(len(gammas))) == 0, "all should be zero"
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

def get_upper_and_lower_bounds(gammas, coefficient_matrix, final_step=10000):
    allowed_aboves = []
    allowed_belows = []
    for i, coefficient_list in enumerate(coefficient_matrix):
        gamma = gammas[i]
        allowed_above, allowed_below = get_difference(gammas, coefficient_list, gamma, final_step=final_step)
        allowed_aboves.append(allowed_above)
        allowed_belows.append(allowed_below)

    return np.array(allowed_aboves), np.array(allowed_belows)




if __name__ == '__main__':
    pass
    # even_spaced = get_even_spacing(0.9, 0.99, 8)[:-1]
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
