# Includes math and visualization for the single gamma bound

import numpy as np
import matplotlib.pyplot as plt

def get_min(v_orig, gamma_orig, gamma_new):
    new_scale = 1 / (1 - gamma_new)
    in_log_numerator = 1 - v_orig + gamma_orig * v_orig
    exponent = np.log(in_log_numerator) / np.log(gamma_orig)
    return new_scale * (1 - gamma_new ** exponent)

def get_max(v_orig, gamma_orig, gamma_new):
    new_scale = 1 / (1 - gamma_new)
    in_log_numerator = v_orig - gamma_orig * v_orig
    exponent = np.log(in_log_numerator) / np.log(gamma_orig)
    return new_scale * (gamma_new ** exponent)


def compare_two_gammas(gamma_orig, gamma_new):
    v_orig = np.linspace(0, 1 / (1 - gamma_orig), 200)
    # min_function = np.vectorize(lambda v: get_min(v, gamma_orig, gamma_new))
    v_new_min = [get_min(v, gamma_orig, gamma_new) for v in v_orig]
    v_new_max = [get_max(v, gamma_orig, gamma_new) for v in v_orig]
    plt.plot(v_orig, v_new_min, label=f"VMIN {gamma_orig} --> {gamma_new}")
    plt.plot(v_orig, v_new_max, label=f"VMAX {gamma_orig} --> {gamma_new}")
    plt.show()

if __name__ == "__main__":
    compare_two_gammas(0.99, 0.991)
