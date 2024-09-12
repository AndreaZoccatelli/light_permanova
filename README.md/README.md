# LightPermanova

A lightweight implementation of PERMANOVA based on Euclidean distance from centroid.

## Overview

One known problem of machine learning models in production which affects their predictive ability is covariate shift. It is defined as a change in the distribution of one or more independent variables used to train the model.

ANOVA is often adopted to assess if two samples are from the same population by comparing the variance of their means (H0: all $$\mu$$’s are equal; H1: at least one pair of $$\mu$$’s are not equal). This test relies, however, on the normality assumption of the samples, which makes it a non-viable solution to effectively monitor batches of data.

PERMANOVA is a multivariate version of ANOVA based on pseudo-F statistic, which makes use of permutations, allowing for a non-parametric estimation.

In the case of covariates shift monitoring the test compares the oiginal sample $$s_0$$ used at time $$t_0$$ to train the model with a new, unseen, sample $$s_1$$ on which the model made predictions at time $$t_1$$.

## Useful links

Read the docs [here](https://light-permanova.readthedocs.io/en/latest/index.html).

This project is part of "[Root.](https://app.gitbook.com/o/Y6GFa7e9eZJJr97DYwE9/s/rv2RagYUGenlarYKGSFe/)".
