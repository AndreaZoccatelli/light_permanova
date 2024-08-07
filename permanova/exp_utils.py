from .permanova_np.lightpermanova import LightPermAnova, PermAnovaSampler
from .permanova_torch.lightpermanova import (
    LightPermAnova as LightPermAnovaTorch,
    PermAnovaSampler as PermAnovaSamplerTorch,
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch


class NumpyTest:
    @staticmethod
    def plot_distributions(sample_1: np.ndarray, sample_2: np.ndarray):
        for i in range(sample_1.shape[1]):
            sns.histplot(sample_1[:, i], label="original")
            sns.histplot(sample_2[:, i], label="new")
            plt.title(f"Difference in var {i}")
            plt.legend()
            plt.show()

    @staticmethod
    def generate_samples_with_normal_covariates(
        dim: int, noise_mean: float, noise_std: float, size: int
    ):
        np.random.seed(6)
        mean1 = np.arange(dim)
        A = np.random.randn(dim, dim)
        covariance_matrix1 = np.dot(A, A.T)

        sample_1 = np.random.multivariate_normal(mean1, covariance_matrix1, size)

        noise = np.random.normal(loc=noise_mean, scale=noise_std, size=sample_1.shape)
        sample_2 = sample_1 + noise

        return sample_1, sample_2

    @staticmethod
    def generate_samples_with_mixture_covariates(
        dim: int, noise_mean_perc: float, noise_std_perc: float, size: int
    ):
        np.random.seed(6)
        sample_1_list = []
        sample_2_list = []
        for _ in range(dim):
            mean1 = np.random.randint(5)
            std1 = np.abs(np.random.randn())
            mean2 = np.random.randint(5)
            std2 = np.abs(np.random.randn())
            # Generate random samples from a uniform distribution to decide which Gaussian to sample from
            mix = np.random.rand(size)

            # Sample from the first Gaussian where mix is less than weight1
            s1 = np.random.normal(loc=mean1, scale=std1, size=size)
            s2 = np.random.normal(loc=mean2, scale=std2, size=size)
            noise = np.random.normal(
                loc=(mean1 * noise_mean_perc),
                scale=(std1 * noise_std_perc),
                size=size,
            )
            noise = np.random.normal(
                loc=(mean2 * noise_mean_perc),
                scale=(std2 * noise_std_perc),
                size=size,
            )
            var_1 = np.where(
                mix < 0.5,
                s1,
                s2,
            )

            var_2 = np.where(
                mix < 0.5,
                s1 + noise,
                s2 + noise,
            )

            var_2 = var_1 + noise
            sample_1_list.append(var_1)
            sample_2_list.append(var_2)
        sample_1 = np.stack(sample_1_list, axis=1)
        sample_2 = np.stack(sample_2_list, axis=1)

        return sample_1, sample_2

    @staticmethod
    def test(sample_1, sample_2, show_plots: bool = False):
        if show_plots:
            NumpyTest.plot_distributions(sample_1, sample_2)

        print("Finding representative subsamples...")
        permanovasampler1 = PermAnovaSampler(sample_1)
        permanovasampler2 = PermAnovaSampler(sample_2)

        sub1 = permanovasampler1.get_representative_sample(tot_permutations=1000)
        sub2 = permanovasampler2.get_representative_sample(tot_permutations=1000)

        print("Computing permutations...")
        permanova = LightPermAnova(sub1, compress=True)
        pvalue = permanova.run_simulation(sub2, tot_permutations=10000)
        print(f"p-value: {round(pvalue,5)}")


class TorchTest:
    @staticmethod
    def plot_distributions(sample_1: torch.Tensor, sample_2: torch.Tensor):
        for i in range(sample_1.shape[1]):
            sns.histplot(sample_1[:, i], label="original")
            sns.histplot(sample_2[:, i], label="new")
            plt.title(f"Difference in var {i}")
            plt.legend()
            plt.show()

    @staticmethod
    def generate_samples_with_normal_covariates(
        dim: int, noise_mean: float, noise_std: float, size: int
    ):
        torch.manual_seed(6)
        mean1 = torch.arange(dim, dtype=torch.float)
        A = torch.randn((dim, dim), dtype=torch.float)
        covariance_matrix1 = torch.mm(A, A.T)

        distribution = torch.distributions.MultivariateNormal(mean1, covariance_matrix1)

        sample_1 = distribution.sample((size,))

        noise = torch.normal(noise_mean, noise_std, sample_1.shape)
        sample_2 = sample_1 + noise

        return sample_1, sample_2

    @staticmethod
    def generate_samples_with_mixture_covariates(
        dim: int, noise_mean_perc: float, noise_std_perc: float, size: int
    ):
        torch.manual_seed(6)
        sample_1 = torch.Tensor([])
        sample_2 = torch.Tensor([])
        for _ in range(dim):
            mean1 = torch.randint(low=0, high=5, size=(1,), dtype=torch.float)
            std1 = torch.abs(torch.randn((1,)))
            mean2 = torch.randint(low=0, high=5, size=(1,), dtype=torch.float)
            std2 = torch.abs(torch.randn((1,)))
            # Generate random samples from a uniform distribution to decide which Gaussian to sample from
            mix = torch.rand((size,))

            # Sample from the first Gaussian where mix is less than weight1
            s1 = torch.normal(mean1.item(), std1.item(), (size,))
            s2 = torch.normal(mean2.item(), std2.item(), (size,))
            noise = torch.normal(
                (mean1.item() * noise_mean_perc),
                (std1.item() * noise_std_perc),
                (size,),
            )
            noise = torch.normal(
                (mean2.item() * noise_mean_perc),
                (std2.item() * noise_std_perc),
                (size,),
            )

            var_1 = torch.where(
                mix < 0.5,
                s1,
                s2,
            )

            var_2 = torch.where(
                mix < 0.5,
                s1 + noise,
                s2 + noise,
            )

            var_2 = var_1 + noise

            sample_1 = torch.cat((sample_1, var_1[:, None]), dim=-1)
            sample_2 = torch.cat((sample_2, var_2[:, None]), dim=-1)

        return sample_1, sample_2

    @staticmethod
    def test(sample_1: torch.Tensor, sample_2: torch.Tensor, show_plots: bool = False):
        if show_plots:
            TorchTest.plot_distributions(sample_1, sample_2)

        print("Finding representative subsamples...")
        permanovasampler1 = PermAnovaSamplerTorch(sample_1)
        permanovasampler2 = PermAnovaSamplerTorch(sample_2)

        sub1 = permanovasampler1.get_representative_sample(tot_permutations=1000)
        sub2 = permanovasampler2.get_representative_sample(tot_permutations=1000)

        print("Computing permutations...")
        permanova = LightPermAnovaTorch(sub1, compress=True)
        pvalue = permanova.run_simulation(sub2, tot_permutations=10000)
        print(f"p-value: {round(pvalue.item(),5)}")
