from ..permanova_np.lightpermanova import LightPERMANOVA
from ..permanova_torch.lightpermanova import LightPERMANOVA as LightPERMANOVATorch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch


class NumpyTest:
    """
    class to perform tabular tests with the NumPy implementation of LightPERMANOVA
    """

    @staticmethod
    def plot_distributions(sample_1: np.ndarray, sample_2: np.ndarray):
        """
        Plots the comparison of the distributions for each variable in the samples.

        Args:
            sample_1 (np.ndarray): original sample
            sample_2 (np.ndarray): sample with noise
        """
        for i in range(sample_1.shape[1]):
            sns.histplot(sample_1[:, i], label="original")
            sns.histplot(sample_2[:, i], label="new")
            plt.title(f"Difference in var {i}")
            plt.legend()
            plt.show()

    @staticmethod
    def generate_samples_with_normal_covariates(
        dim: int, noise_mean: float, noise_std: float, size: int
    ) -> tuple:
        """
        Generates samples with normally distributed covariates

        Args:
            dim (int): number of covariates that describe the samples
            noise_mean (float): mean of the noise applied to each variable of the original sample
            noise_std (float): std of the noise applied to each variable of the original sample
            size (int): size of the samples

        Returns:
            tuple: (sample_1: np.ndarray, sample_2: np.ndarray)
        """
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
        dim: int, noise_mean_factor: float, noise_std_factor: float, size: int
    ) -> tuple:
        """
        Generates samples with covariates drawn from mixture of two gaussians

        Args:
            dim (int): number of covariates that describe the samples
            noise_mean_factor (float): to each of the gaussians of sample_2 will be applied a noise with mean = noise_mean_factor * original mean of the gaussian
            noise_std_factor (float): to each of the gaussians of sample_2 will be applied a noise with std = noise_std_factor * original std of the gaussian
            size (int): size of the samples

        Returns:
            tuple: (sample_1: np.ndarray, sample_2: np.ndarray)
        """
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
            noise_1 = np.random.normal(
                loc=(mean1 * noise_mean_factor),
                scale=(std1 * noise_std_factor),
                size=size,
            )
            noise_2 = np.random.normal(
                loc=(mean2 * noise_mean_factor),
                scale=(std2 * noise_std_factor),
                size=size,
            )
            var_1 = np.where(
                mix < 0.5,
                s1,
                s2,
            )

            var_2 = np.where(
                mix < 0.5,
                s1 + noise_1,
                s2 + noise_2,
            )

            sample_1_list.append(var_1)
            sample_2_list.append(var_2)
        sample_1 = np.stack(sample_1_list, axis=1)
        sample_2 = np.stack(sample_2_list, axis=1)

        return sample_1, sample_2

    @staticmethod
    def test(sample_1, sample_2, show_plots: bool = False):
        if show_plots:
            NumpyTest.plot_distributions(sample_1, sample_2)

        print("Computing permutations...")
        permanova = LightPERMANOVA(sample_1, compress=True)
        pvalue = permanova.run_simulation(sample_2, tot_permutations=10000)
        print(f"p-value: {round(pvalue,5)}")


class TorchTest:
    """
    class to perform tabular tests with the PyTorch implementation of LightPERMANOVA
    """

    @staticmethod
    def plot_distributions(sample_1: torch.Tensor, sample_2: torch.Tensor):
        """
        Plots the comparison of the distributions for each variable in the samples.

        Args:
            sample_1 (np.ndarray): original sample
            sample_2 (np.ndarray): sample with noise
        """
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
        """
        Generates samples with normally distributed covariates

        Args:
            dim (int): number of covariates that describe the samples
            noise_mean (float): mean of the noise applied to each variable of the original sample
            noise_std (float): std of the noise applied to each variable of the original sample
            size (int): size of the samples

        Returns:
            tuple: (sample_1: np.ndarray, sample_2: np.ndarray)
        """
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
        dim: int, noise_mean_factor: float, noise_std_factor: float, size: int
    ):
        """
        Generates samples with covariates drawn from mixture of two gaussians

        Args:
            dim (int): number of covariates that describe the samples
            noise_mean_factor (float): to each of the gaussians of sample_2 will be applied a noise with mean = noise_mean_factor * original mean of the gaussian
            noise_std_factor (float): to each of the gaussians of sample_2 will be applied a noise with std = noise_std_factor * original std of the gaussian
            size (int): size of the samples

        Returns:
            tuple: (sample_1: np.ndarray, sample_2: np.ndarray)
        """
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
            noise_1 = torch.normal(
                (mean1.item() * noise_mean_factor),
                (std1.item() * noise_std_factor),
                (size,),
            )
            noise_2 = torch.normal(
                (mean2.item() * noise_mean_factor),
                (std2.item() * noise_std_factor),
                (size,),
            )

            var_1 = torch.where(
                mix < 0.5,
                s1,
                s2,
            )

            var_2 = torch.where(
                mix < 0.5,
                s1 + noise_1,
                s2 + noise_2,
            )

            sample_1 = torch.cat((sample_1, var_1[:, None]), dim=-1)
            sample_2 = torch.cat((sample_2, var_2[:, None]), dim=-1)

        return sample_1, sample_2

    @staticmethod
    def test(sample_1: torch.Tensor, sample_2: torch.Tensor, show_plots: bool = False):
        if show_plots:
            TorchTest.plot_distributions(sample_1, sample_2)

        print("Computing permutations...")
        permanova = LightPERMANOVATorch(sample_1, compress=True)
        pvalue = permanova.run_simulation(sample_2, tot_permutations=10000)
        print(f"p-value: {round(pvalue.item(),5)}")
