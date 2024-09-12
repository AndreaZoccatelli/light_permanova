# Overview

## PERMANOVA for covariate shift monitoring
One known problem of machine learning models in production which affects their predictive ability is covariate shift. It is defined as a change in the distribution of one or more independent variables used to train the model.

ANOVA is often adopted to assess if two samples are from the same population by comparing the variance of their means (H0: all $\mu$’s are equal; H1: at least one pair of $\mu$’s are
not equal). This test relies, however, on the normality assumption of the samples, which makes it a non-viable solution to effectively monitor batches of data.

PERMANOVA is a multivariate version of ANOVA based on pseudo-F statistic, which makes use of permutations, allowing for a non-parametric estimation. 

In the case of covariates shift monitoring the test compares the oiginal sample $s_0$ used at time $t_0$ to train the model with a new, unseen, sample $s_1$ on which the model made predictions at time $t_1$.

The test starts by computing the pseudo-F statistics on the two samples as follows:

$$F = \frac{\frac{SS_B}{p-1}}{\frac{SS_W}{N-p}},$$

where $p$ is the number of groups (in this case $p = 2$), $n$ is the number of observation in a group and $N$ is the total number of observations.

$SS_W$ is the sum of squares within the groups (where $\delta _{ij}$ is 1 if the observations i and j belong to the same group, and 0 otherwise):

$$SS_W = \frac{1}{n}\sum_{i=1}^{N-1}\sum_{j=i+1}^{N}d^2_{ij}\delta_{ij}$$

and $SS_B$ is the sum of squares between the groups:

$$SS_B = SS_T - SS_W$$

$SS_T$ is the total sum of squares:

$$SS_T = \frac{1}{N}\sum_{i=1}^{N-1}\sum_{j=i+1}^{N}d^2_{ij}$$
 

Then for each permutation $\pi$ items are shuffled between each group and respective $F^\pi$ is computed. The p-value is then computed as follows:

$$p = \frac{\text{number of permutations with} \space F^{\pi} \geq F}{\text{total number of permutations}}$$

if p-value is lower than the chosen significance level $\alpha$ then the null hypothesis is rejected, so the variance of the two groups differ. This can be used as a first alert to further analyse the new sample and evaluate if a model retrain could be necessary. 

## A "lightweight" implementation of PERMANOVA
The usual implementation of PERMANOVA relies on the distance matrix between every observation in the samples, this results in a computationally expensive test. 

However, if the observations are considered as points in the Euclidean space it is possible to revise the provious formulas leveraging distances from centroids.

$SS_T$ = the sum of squared distances from the observations to the overall centroid

$SS_W$ = the sum of squared distances from the observations to their own group centroid

This allows to reduce the time-complexity of the algorithm from $O(N^2)$ to $O(N)$, at the expense of relying on Euclidean distance. If another metric could be more suitable to the variables that describe the samples, the complete distance matrix should be adopted.

## References
<a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1442-9993.2001.01070.pp.x" target="_blank">Research paper</a>

<a href="https://learninghub.primer-e.com/books/permanova-for-primer-guide-to-software-and-statistical-methods/chapter/chapter-1-permutational-anova-and-manova-permanova" target="_blank">PRIMER</a>

<a href="https://www.jwilber.me/permutationtest/" target="_blank">Intuitive explanation of permutation tests</a>