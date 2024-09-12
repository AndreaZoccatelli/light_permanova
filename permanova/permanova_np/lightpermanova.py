import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import warnings


class LightPERMANOVA:
    """
    Class to perform `PERMANOVA <https://en.wikipedia.org/wiki/Permutational_analysis_of_variance>`_
    analysis between two samples.
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

    def _get_centroids(self, sample_1: np.ndarray, sample_2: np.ndarray) -> tuple:
        """
        LightPermanova makes use of distance from samples' centroids instead of
        a complete distance matrix for faster computation.

        Returns:
            tuple: a tuple of np.ndarray, (centroid_1, centroid_2)
        """
        temp_centroid_1 = np.mean(sample_1, axis=0)
        temp_centroid_2 = np.mean(sample_2, axis=0)
        if not hasattr(self, "centroid_complete"):
            self.centroid_complete = np.mean(self.complete_sample, axis=0)
        return temp_centroid_1, temp_centroid_2

    def _get_sst(self) -> float:
        """
        Method to compute the total sum of squares (SST)

        Returns:
            float: total sum of squares
        """
        distances = np.linalg.norm(
            self.complete_sample - self.centroid_complete, axis=1
        )
        self.sst = np.sum(distances**2)  # fix

    def _get_ssw(
        self,
        sample_1: np.ndarray,
        temp_centroid_1: np.ndarray,
        sample_2: np.ndarray,
        temp_centroid_2: np.ndarray,
    ) -> float:
        """
        Method to compute the within-group sum of squares (SSW)

        Returns:
            float: within-group sum of squares
        """
        distances_1 = np.linalg.norm(sample_1 - temp_centroid_1, axis=1)
        distances_2 = np.linalg.norm(sample_2 - temp_centroid_2, axis=1)
        ss1 = np.sum(distances_1**2)
        ss2 = np.sum(distances_2**2)
        return ss1 + ss2

    def _get_pseudo_f(self, sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Method to compute [pseudo-F statistic](https://learninghub.primer-e.com/books/permanova-for-primer-guide-to-software-and-statistical-methods/page/15-the-pseudo-f-statistic).

        Returns:
            float: pseudo-F statistic
        """
        temp_centroid_1, temp_centroid_2 = self._get_centroids(sample_1, sample_2)
        if not hasattr(self, "sst"):
            self._get_sst()
        ssw = self._get_ssw(sample_1, temp_centroid_1, sample_2, temp_centroid_2)
        ssb = self.sst - ssw
        return ssb / (ssw / self.dof)

    def _stratified_sample(
        self,
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        sample_1_share: float,
        sample_2_share: float,
    ) -> np.ndarray:
        """
        Create a stratified sample by combining subsets of two datasets.

        Parameters:
            sample_1: The first dataset to sample from.
            sample_2: The second dataset to sample from.
            sample_1_share: Proportion of the first dataset to sample.
            sample_2_share: Proportion of the second dataset to sample.

        Returns:
            A numpy array containing the combined stratified sample.
        """
        sample_i11 = np.random.choice(
            sample_1.shape[0],
            round(sample_1_share * sample_1.shape[0]),
            replace=False,
        )
        sample_i12 = np.random.choice(
            sample_2.shape[0],
            round(sample_2_share * sample_1.shape[0]),
            replace=False,
        )
        permuted_sample_1 = np.concatenate(
            [sample_1[sample_i11, :], sample_2[sample_i12, :]]
        )
        sample_i21 = np.setdiff1d(range(sample_1.shape[0]), sample_i11)
        sample_i22 = np.setdiff1d(range(sample_2.shape[0]), sample_i12)

        permuted_sample_2 = np.concatenate(
            [sample_1[sample_i21, :], sample_2[sample_i22, :]]
        )
        return permuted_sample_1, permuted_sample_2

    def _permute(self):
        permuted_sample_1, permuted_sample_2 = self._stratified_sample(
            self.sample_1, self.sample_2, self.sample_1_share, self.sample_2_share
        )
        pseudo_f = self._get_pseudo_f(permuted_sample_1, permuted_sample_2)
        return pseudo_f

    def run_simulation(self, new_sample: np.ndarray, tot_permutations: int) -> float:
        """
        Args:
            new_sample (np.ndarray): sample that will be compared with sample_1
            tot_permutations (int): total number of permutations

        Returns:
            float: p_value (null hypothesis is sample_1 and new_sample are not different)
        """
        self._setup(new_sample)
        self.starting_pseudo_f = self._get_pseudo_f(
            sample_1=self.sample_1, sample_2=self.sample_2
        )
        self.over = 0
        self.pseudo_f_values = []
        self.pseudo_f_values = Parallel(n_jobs=-1)(
            delayed(self._permute)() for _ in tqdm(range(tot_permutations))
        )
        self.pseudo_f_values = np.array(self.pseudo_f_values)
        self.over = np.sum(self.pseudo_f_values > self.starting_pseudo_f)
        return (self.over + 1) / (tot_permutations + 1)
