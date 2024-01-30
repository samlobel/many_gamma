import torch
from torch import nn
import numpy as np
from gamma_utilities import *
import time


# Okay, how will we do this. We'll set up a torch optimization equation is all. Should be simple.
# Pass in a list of gammas you track, and a list of gammas you're trying to optimize for. Should
# return matrix of coefficients.

class CoefficientsModule(nn.Module):
    # I think for this its okay to say we're approximating the same gammas as we know 
    # Let's make optimization its own thing.
    def __init__(self, gammas, regularization=0., skip_self_map=False, only_from_lower=False):
        super().__init__()
        self.gammas = torch.Tensor(gammas)
        assert torch.allclose(self.gammas, torch.sort(self.gammas)[0])
        # self.gammas_to_approximate = torch.Tensor(gammas_to_approximate)
        self.regularization = regularization
        self._skip_self_map = skip_self_map
        self._only_from_lower = only_from_lower # upper diagonal
        # TODO: Try out different initializations perhaps.
        # To start, initialize to L2 solution.
        perfect_l2_coefficients = get_constraint_matrix(gammas, regularization=regularization, skip_self_map=skip_self_map, only_from_lower=only_from_lower)
        perfect_l2_coefficients = perfect_l2_coefficients.T # Need to transpose it sadly. TODO: Make correct off bat?
        perfect_l2_coefficients = perfect_l2_coefficients.astype(np.float32)
        self._unzeroed_coefficients = nn.Parameter(torch.tensor(perfect_l2_coefficients)) # Zeroed comes later.

    @property
    def coefficients(self):
        # Ugh, remember that its transposed. So now the first column is almost all zeros, not the first row.
        if self._skip_self_map and self._only_from_lower:
                return torch.triu(self._unzeroed_coefficients, 1)
        elif self._skip_self_map and not self._only_from_lower:
                return self._unzeroed_coefficients * (1 - torch.eye(len(self.gammas)))
        elif not self._skip_self_map and self._only_from_lower:
            return torch.triu(self._unzeroed_coefficients, 0)
        elif not self._skip_self_map and not self._only_from_lower:
            return self._unzeroed_coefficients
        else:
            raise Exception("Shouldn't be here.")

    def get_coefficients(self):
        return self.coefficients.detach().numpy()

    def solve(self, num_steps=10000, lr=0.001, metric="abs", out_until=10000):
        assert metric == "abs", "No real reason to do anything else at the moment, considering we can get an exact solution the other way."
        # assert metric in ["abs", "square"], f"Metric must be abs or square but got {metric}"
        # I want roughly the same answer if I double the density.
        # I think that's actually tough, because when you double the number of things
        # it'll put half weights on each which means the L2 goes down. How about I just do sum
        # like before.
        start_time = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        gammas = self.gammas[None, ...].repeat(out_until, 1)
        powers = torch.arange(0, out_until)[..., None]
        gammas_over_time = gammas ** powers
        total_abs_differences, biggest_abs_differences, coefficient_magnitudes, total_losses = [], [], [], []
        # import ipdb; ipdb.set_trace()
        for i in range(num_steps):
            optimizer.zero_grad()
            computed_over_time = torch.matmul(gammas_over_time, self.coefficients)

            difference_per_gamma = torch.abs(computed_over_time - gammas_over_time).sum(axis=0)
            difference_total = difference_per_gamma.sum()
            coefficient_magnitude = (self.coefficients ** 2).sum()
            loss = difference_total + (coefficient_magnitude * self.regularization)
            loss.backward()
            optimizer.step()
            total_abs_differences.append(difference_total.detach().item())
            biggest_abs_differences.append(difference_per_gamma.max().detach().item())
            coefficient_magnitudes.append(coefficient_magnitude.detach().item())
            total_losses.append(loss.detach().item())
            if i % 1000 == 0:
                print(f"Step {i} Loss: {loss.item():.5f} Tota Diff: {difference_total.detach().item():.5f} Biggest Diff: {difference_per_gamma.max().detach().item():.5f} Coefficient Magnitude: {coefficient_magnitude.detach().item():.5f}")

        end_time = time.time()
        print(f"Time to do {num_steps} coefficient optimization steps: {end_time - start_time:.4f}")
        return {
            "total_abs_differences": total_abs_differences,
            'biggest_abs_differences': biggest_abs_differences,
            "coefficient_magnitudes": coefficient_magnitudes,
            "total_losses": total_losses,
        }

a = None
# Seems like I'm doing something wrong if its not getting any closer. Probably something wrong here
# Seems to learn the lowest one just fine but not the others. Must have messed up in there somewhere.
# Maybe I try to write it a little more cleanly or whatever.
# The best case for this is a set of coefficients that's a nicer match than the L2 way.

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    gammas = get_even_spacing(0.9, 0.997, 50)
    regularization = 1.
    trainer = CoefficientsModule(gammas, regularization=regularization, skip_self_map=True)
    # log_dict = trainer.solve(10001, lr=0.0001)
    for lr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        log_dict = trainer.solve(4001, lr=lr)
        plt.plot(log_dict['total_abs_differences'])
        plt.yscale("log")
        plt.title(f"lr: {lr}")
        plt.show()


                   

