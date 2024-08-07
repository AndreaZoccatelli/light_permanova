import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import warnings


class PermAnova:
    """
    Class to perform [PERMANOVA](https://en.wikipedia.org/wiki/Permutational_analysis_of_variance) analysis between two samples.
    """

    def _pca_compress(self):
        """
        Defines a PCA with a number of components that explain more than 95% of variance in sample 1.
        """
        self.pca = PCA(n_components=0.95)
        self.sample_1 = self.pca.fit_transform(self.sample_1)

    def __init__(self, sample_1: np.ndarray, compress: bool = True) -> None:
        """
        Args:
            sample_1 (np.ndarray): the original sample.
            compress (bool, optional): If True both sample_1 and sample_2 will be compressed using PCA. Defaults to True.
        """
        self.sample_1 = sample_1
        self.compress = compress
        if self.compress:
            self._pca_compress()

    def _check_size(self):
        """
        Checks if one of the two samples is more than 10 times smaller than the other, important for test reliability (permutations are based on stratified sampling)
        """
        if self.sample_2.shape[0] / self.sample_1.shape[0] < 0.1:
            warnings.warn(
                "New sample is more than 10 times smaller than original sample, consider subsampling original sample for a more reliable test."
            )
        elif self.sample_1.shape[0] / self.sample_2.shape[0] < 0.1:
            warnings.warn(
                "Original sample is more than 10 times smaller than new sample, consider subsampling new sample for a more reliable test."
            )
        self.sample_2_share = self.sample_2.shape[0] / (
            self.sample_1.shape[0] + self.sample_2.shape[0]
        )
        self.sample_1_share = 1 - self.sample_2_share

    def _setup(self, new_sample: np.ndarray):
        if self.compress:
            self.sample_2 = self.pca.transform(new_sample)
        else:
            self.sample_2 = new_sample
        self._check_size()
        self.complete_sample = np.concatenate([self.sample_1, self.sample_2])
        self.dof = self.complete_sample.shape[0] - 2
        self.split = len(self.complete_sample) // 2

    def _get_distance_matrix(self) -> np.ndarray:
        """
        Method to calculate distance matrix for the initial samples
        """
        complete_sample = self.complete_sample
        self.distance_matrix = complete_sample[:, np.newaxis, :] - complete_sample
        self.distance_matrix = np.linalg.norm(self.distance_matrix, axis=2)

    def _filter_distance_matrix(self, i1: np.array, i2: np.array = None) -> np.array:
        if i2 is not None:
            index = np.concatenate([i1, i2])
            filtered_distance_matrix = self.distance_matrix[index, :]
            mask = np.zeros(
                (len(filtered_distance_matrix), filtered_distance_matrix.shape[1]),
                dtype=bool,
            )
            mask[:, index] = True
            mask[range(len(filtered_distance_matrix)), index] = False

        else:
            filtered_distance_matrix = self.distance_matrix
            mask = ~np.eye(len(filtered_distance_matrix), dtype=bool)

        filtered_distance_matrix = filtered_distance_matrix[mask].reshape(
            len(filtered_distance_matrix), len(filtered_distance_matrix) - 1
        )
        return filtered_distance_matrix

    def _get_sst(self, distance_matrix: np.ndarray) -> float:
        """
        Method to compute the total sum of squares (SST)

        Returns:
            float: total sum of squares
        """
        return np.mean(np.sum(distance_matrix**2, axis=1))

    def _get_ssw(self, distance_matrix, i1) -> float:
        """
        Method to compute the within-group sum of squares (SSW)

        Returns:
            float: within-group sum of squares
        """
        ss1 = np.mean(
            np.sum(
                distance_matrix[: len(i1), : len(i1) - 1] ** 2,
                axis=1,
            )
        )
        ss2 = np.mean(
            np.sum(
                distance_matrix[len(i1) :, len(i1) :] ** 2,
                axis=1,
            )
        )
        return ss1 + ss2

    def _get_pseudo_f(self, i1: np.array, i2: np.array = None) -> float:
        """
        Method to compute [pseudo-F statistic](https://learninghub.primer-e.com/books/permanova-for-primer-guide-to-software-and-statistical-methods/page/15-the-pseudo-f-statistic).

        Returns:
            float: pseudo-F statistic
        """
        distance_matrix = self._filter_distance_matrix(i1, i2)
        sst = self._get_sst(distance_matrix)
        ssw = self._get_ssw(distance_matrix, i1)
        ssb = sst - ssw
        return ssb / (ssw / (self.complete_sample.shape[0] - 2))

    def _stratified_sample(
        self,
        sample_1: np.ndarray,
        sample_1_share: float,
        sample_2_share: float,
    ) -> tuple:
        """
        Create a stratified sample by combining subsets of two datasets.

        Parameters:
            sample_1: The first dataset to sample from.
            sample_1_share: Proportion of the first dataset to sample.
            sample_2_share: Proportion of the second dataset to sample.

        Returns:
            A tuple with the permuted indices for the two samples.
        """
        sample_i11 = np.random.choice(
            sample_1.shape[0] - 1,
            round(sample_1_share * sample_1.shape[0]),
            replace=False,
        )
        sample_i12 = np.random.choice(
            range(sample_1.shape[0], len(self.index)),
            round(sample_2_share * sample_1.shape[0]),
            replace=False,
        )
        i1 = np.concatenate([sample_i11, sample_i12])
        i2 = np.setdiff1d(range(self.index.shape[0]), i1)
        return i1, i2

    def _permute(self):
        i1, i2 = self._stratified_sample(
            self.sample_1, self.sample_1_share, self.sample_2_share
        )
        pseudo_f = self._get_pseudo_f(i1, i2)
        return pseudo_f

    def run_simulation(self, new_sample: np.ndarray, tot_permutations: int) -> float:
        self._setup(new_sample)
        self._get_distance_matrix()
        self.starting_pseudo_f = self._get_pseudo_f(i1=range(self.sample_1.shape[0]))
        self.over = 0
        self.pseudo_f_values = []
        self.index = np.arange(len(self.complete_sample))
        self.pseudo_f_values = Parallel(n_jobs=-1)(
            delayed(self._permute)() for _ in tqdm(range(tot_permutations))
        )
        self.pseudo_f_values = np.array(self.pseudo_f_values)
        self.over = np.sum(self.pseudo_f_values > self.starting_pseudo_f)
        return (self.over + 1) / (tot_permutations + 1)
