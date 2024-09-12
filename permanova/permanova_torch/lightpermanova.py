import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import warnings


class LightPERMANOVA:
    """
    Class to perform PERMANOVA between two samples.
    """

    def _pca_compress(self):
        """
        Defines a PCA with a number of components that explain more than 95% of variance in sample 1.
        """
        self.pca = PCA(n_components=0.95)
        self.sample_1 = self.pca.fit_transform(self.sample_1)
        self.sample_1 = torch.from_numpy(self.sample_1)

    def __init__(self, sample_1: torch.Tensor, compress: bool = True) -> None:
        """
        Args:
            sample_1 (torch.Tensor): the original sample.
            compress (bool, optional): If True both sample_1 and sample_2 will be compressed using PCA. Defaults to True.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_1 = sample_1
        self.compress = compress
        if self.compress:
            self._pca_compress()
        self.sample_1 = self.sample_1.to(self.device)

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

    def _setup(self, new_sample: torch.Tensor):
        if self.compress:
            self.sample_2 = self.pca.transform(new_sample)
            self.sample_2 = torch.from_numpy(self.sample_2)
        else:
            self.sample_2 = new_sample

        self.sample_2 = self.sample_2.to(self.device)
        self._check_size()
        self.complete_sample = torch.concatenate([self.sample_1, self.sample_2])
        self.dof = self.complete_sample.shape[0] - 2
        self.split = len(self.complete_sample) // 2

    def _get_centroids(self, sample_1: torch.Tensor, sample_2: torch.Tensor) -> tuple:
        """
        LightPermanova makes use of distance from samples' centroids instead of
        a complete distance matrix for faster computation.

        Returns:
            tuple: a tuple of torch.Tensor, (centroid_1, centroid_2)
        """
        temp_centroid_1 = torch.mean(sample_1, axis=0)
        temp_centroid_2 = torch.mean(sample_2, axis=0)
        if not hasattr(self, "centroid_complete"):
            self.centroid_complete = torch.mean(self.complete_sample, axis=0)
        return temp_centroid_1, temp_centroid_2

    def _get_sst(self) -> float:
        """
        Method to compute the total sum of squares (SST)

        Returns:
            float: total sum of squares
        """
        distances = torch.linalg.norm(
            self.complete_sample - self.centroid_complete, axis=1
        )
        self.sst = torch.sum(distances**2)  # fix

    def _get_ssw(
        self,
        sample_1: torch.Tensor,
        temp_centroid_1: torch.Tensor,
        sample_2: torch.Tensor,
        temp_centroid_2: torch.Tensor,
    ) -> float:
        """
        Method to compute the within-group sum of squares (SSW)

        Returns:
            float: within-group sum of squares
        """
        distances_1 = torch.linalg.norm(sample_1 - temp_centroid_1, axis=1)
        distances_2 = torch.linalg.norm(sample_2 - temp_centroid_2, axis=1)
        ss1 = torch.sum(distances_1**2)
        ss2 = torch.sum(distances_2**2)
        return ss1 + ss2

    def _get_pseudo_f(self, sample_1: torch.Tensor, sample_2: torch.Tensor) -> float:
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
        sample_1: torch.Tensor,
        sample_2: torch.Tensor,
        sample_1_share: float,
        sample_2_share: float,
    ) -> torch.Tensor:
        """
        Create a stratified sample by combining subsets of two datasets.

        Parameters:
        - sample_1: The first dataset to sample from.
        - sample_2: The second dataset to sample from.
        - sample_1_share: Proportion of the first dataset to sample.
        - sample_2_share: Proportion of the second dataset to sample.

        Returns:
        - A numpy array containing the combined stratified sample.
        """
        permuted_i1 = torch.randperm(sample_1.shape[0])
        size_1 = round(sample_1_share * sample_1.shape[0])
        sample_i11 = permuted_i1[:size_1]

        permuted_i2 = torch.randperm(sample_2.shape[0])
        size_2 = round(sample_2_share * sample_2.shape[0])
        sample_i12 = permuted_i2[:size_2]

        permuted_sample_1 = torch.concatenate(
            [sample_1[sample_i11, :], sample_2[sample_i12, :]]
        )

        sample_i21 = permuted_i1[size_1:]
        sample_i22 = permuted_i2[size_2:]

        permuted_sample_2 = torch.concatenate(
            [sample_1[sample_i21, :], sample_2[sample_i22, :]]
        )
        return permuted_sample_1, permuted_sample_2

    def _get_stratified_samples(self) -> tuple:
        """
        Get two stratified samples by applying different proportions.

        Returns:
        - A tuple of two numpy arrays containing the stratified samples.
        """
        permuted_sample_1 = self._stratified_sample(
            self.sample_1, self.sample_2, self.sample_1_share, self.sample_2_share
        )
        permuted_sample_2 = self._stratified_sample(
            self.sample_2, self.sample_1, self.sample_2_share, self.sample_1_share
        )
        return permuted_sample_1, permuted_sample_2

    def _permute(self):
        permuted_sample_1, permuted_sample_2 = self._stratified_sample(
            self.sample_1, self.sample_2, self.sample_1_share, self.sample_2_share
        )
        pseudo_f = self._get_pseudo_f(permuted_sample_1, permuted_sample_2)
        return pseudo_f

    def run_simulation(self, new_sample: torch.Tensor, tot_permutations: int) -> float:
        """
        Args:
            new_sample (torch.Tensor): sample that will be compared with sample_1
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
        self.pseudo_f_values = torch.Tensor(self.pseudo_f_values).to(self.device)
        self.over = torch.sum(self.pseudo_f_values > self.starting_pseudo_f)
        return (self.over + 1) / (tot_permutations + 1)
