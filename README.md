# pcarpet
[![Build Status](https://travis-ci.org/niksirbi/pcarpet.svg?branch=master)](https://travis-ci.org/niksirbi/pcarpet)

<p align="center">
  <img src="images/logo.png" width="400" />
</p>

'pcarpet' is a small python package that allows you to create a **carpet plot** from fMRI data and decompose it with **PCA**.

A 'carpet plot' here refers to a 2d representation of fMRI data (voxels x time), 
very similar to 'The Plot' described by Jonathan D Power ([Power 2017](https://www.sciencedirect.com/science/article/abs/pii/S1053811916303871?via%3Dihub)). 
It has been referred to with various other names, including 'grayplot'. 
This visual representation of fMRI data is suited for identifying wide-spread signal fluctutations 
([Aquino et al. 2020](https://www.sciencedirect.com/science/article/pii/S1053811920301014)), 
which often come from non-neural sources (e.g. head motion).

That said, the carpet plot can also reveal 'real' neural activity, especially when the activity is
very slow and synchronous, as is the case for anesthesia-induced burst-suppression (Sirmpiltze et al. 2021).
The `pcarpet` package implements the analytical pipeline used in the Sirmpiltze et al. 2021 paper.
The pipeline consists of the following steps:

1. Generating a carpet plot within a given region-of-interest (e.g cortex), defined by a mask. To make wide-spread fluctuations more visually prominent, the voxel time-series (carpet rows) are normalized (z-score) and re-ordered according to their correlation with the mean time-series (across voxels).
2. Applying Principal Component Analysis (PCA) to the carpet matrix (using the `scikit-learn` implementation), and extracting a given number (`ncomp`) of first Principal Components - hereafter referred to as 'fPCs'. The fPCs represent the temporal patterns of activity that explain most of the variance.
3. Correlating the 'fPCs' with all voxel time-courses within the carpet, and thus getting a distribution of correlation coefficients per fPC.
4. Correlating the above fPCs with the entire fMRI scan (including areas outside the mask), to get the brain-wide spatial distribution of each fPC.
5. Providing a visual summary of the results (example below).


<p align="center">
  <img src="images/visual_report.png" width="800" />
</p>

This project was created using the [shablona template](https://github.com/uwescience/shablona).
