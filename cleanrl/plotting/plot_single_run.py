from comparison_plotting_utils import *
from plotting_utils import *


def plot_single_run(run_directory, field_to_plot, smoothen=10, log_scale=False):
    pkl_path = os.path.join(run_directory, 'log_dict.pkl')
    frames, quantities_to_plot = get_summary_data(pkl_path, field_to_plot)
    print(quantities_to_plot)
    # Might need to add dummy dimension
    frames = np.array(frames)
    quantities_to_plot = np.array(quantities_to_plot)
    quantities_to_plot = quantities_to_plot[None, :]
    generate_plot(frames, quantities_to_plot, field_to_plot, smoothen=smoothen)
    if log_scale:
        plt.yscale('log')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    """Set up so that we can do a single run plot"""
    # experiment_name = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/tabular/pairwise_ring_2/"
    experiment_name = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/tabular/single_runs/fixed_true_init"
    # experiment_name = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/mcar/single_runs/fixed_true_init"    
    run_title = "fixed_memory_thing"
    run_directory = os.path.join(experiment_name, run_title)
    # field_to_plot = "tabular_total_mse_from_optimal"
    # field_to_plot = "td_loss"
    field_to_plot = "tabular_largest_gamma_mse_from_optimal"
    # field_to_plot = "last_gamma_td_loss"

    plot_single_run(run_directory, field_to_plot=field_to_plot, smoothen=10, log_scale=True)

