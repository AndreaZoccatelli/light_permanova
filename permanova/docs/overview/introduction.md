# PERMANOVA for covariate shift monitoring

One known problem of machine learning models in production which affects their predictive ability is covariate shift. It is defined as a change in the distribution of one or more independent variables used to train the model.

ANOVA is often adopted to assess if two samples are from the same population by comparing the variance of their means (H0: all µ’s are equal; H1: at least one pair of µ’s are
not equal). This test relies, however, on the normality assumption of the samples, which makes it a non-viable solution to effectively monitor batches of data.

PERMANOVA is a multivariate version of ANOVA based on Pseudo-F statistic, which makes use of permutations, allowing for a non-parametric estimation. 

$$F = \frac{\frac{SS_A}{p-1}}{\frac{SS_W}{N-p}}$$

$$p = \frac{\text{number of permutations with} \space F^{\pi} \geq F}{\text{total number of permutations}}$$