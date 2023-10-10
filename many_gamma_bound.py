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

def test_compute_coefficients_same():
    assert np.allclose(compute_coefficients([0.99], 0.99), [1, ]), compute_coefficients([0.99], 0.99)
    assert np.allclose(compute_coefficients([0.9], 0.9), [1, ]), compute_coefficients([0.9], 0.9)

def make_comparison_plot(known_gammas, gamma_target, coefficients=None, save_path=None):
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

    title = fr"-{neg_diff:.4f} < $\Delta$V < {pos_diff:.4f}"
    plt.plot(x_axis, v_orig, label="Original")
    plt.plot(x_axis, v_combined, label="Combined")
    plt.title(title)
    plt.legend()

    # caption_string = f"Known gammas: {known_gammas}\nTarget gamma: {gamma_target}"
    caption_string = f"Known gammas: {known_gammas}\nTarget gamma: {gamma_target}\n -{neg_diff:.4f} < Delta V < {pos_diff:.4f}"
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

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    

def get_difference(known_gammas, coefficients, gamma_target, final_step = 1000):
    # Numerically integrates upper and lower difference bounds.
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




if __name__ == '__main__':
    # make_comparison_plot([0.9, 0.95], 0.8)
    # make_comparison_plot([0.85, 0.95], 0.9)
    make_comparison_plot([0.88, 0.92], 0.9)
    # make_comparison_plot(np.linspace(0.9, 0.99, 10), 0.8)
    # make_comparison_plot([0.9, 0.95], 0.9)
    # make_comparison_plot([0.79, 0.81], 0.8)
