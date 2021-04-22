from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
import scipy.optimize as opt
from scipy.special import erf
from sklearn.decomposition import PCA

__all__ = ["Model", "Fit", "opt_err_func", "transform_data", "cumgauss"]

# Small value added to some denominators to avoid zero division
EPSILON = 1e-9


def pearsonr_2d(A, B):
    """Calculate row-wise Pearson's correlation between 2 2d-arrays

    Parameters
    ----------
    A : 2d-array
        shape N x T
    B : 2d-array
        shape M x T
    Returns
    -------
    R : 2d-array
        N x M shaped correlation matrix between all row combinations of A and B
    """

    #  Subtract row-wise mean from input arrays
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get and return correlation coefficient
    numerator = np.dot(A_mA, B_mB.T)
    denominator = np.sqrt(np.dot(ssA[:, None], ssB[None])) + EPSILON
    return numerator / denominator


def transform_data(data):
    """
    Function that takes experimental data and gives us the
    dependent/independent variables for analysis.

    Parameters
    ----------
    data : Pandas DataFrame or string.
        If this is a DataFrame, it should have the columns `contrast1` and
        `answer` from which the dependent and independent variables will be
        extracted. If this is a string, it should be the full path to a csv
        file that contains data that can be read into a DataFrame with this
        specification.

    Returns
    -------
    x : array
        The unique contrast differences.
    y : array
        The proportion of '2' answers in each contrast difference
    n : array
        The number of trials in each x,y condition
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    contrast1 = data['contrast1']
    answers = data['answer']

    x = np.unique(contrast1)
    y = []
    n = []

    for c in x:
        idx = np.where(contrast1 == c)
        n.append(float(len(idx[0])))
        answer1 = len(np.where(answers[idx[0]] == 1)[0])
        y.append(answer1 / n[-1])
    return x, y, n


def cumgauss(x, mu, sigma):
    """
    The cumulative Gaussian at x, for the distribution with mean mu and
    standard deviation sigma.

    Parameters
    ----------
    x : float or array
       The values of x over which to evaluate the cumulative Gaussian function

    mu : float
       The mean parameter. Determines the x value at which the y value is 0.5

    sigma : float
       The variance parameter. Determines the slope of the curve at the point
       of Deflection

    Returns
    -------

    g : float or array
        The cumulative gaussian with mean $\\mu$ and variance $\\sigma$
        evaluated at all points in `x`.

    Notes
    -----
    Based on:
    http://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function

    The cumulative Gaussian function is defined as:

    .. math::

        \\Phi(x) = \\frac{1}{2} [1 + erf(\\frac{x}{\\sqrt{2}})]

    Where, $erf$, the error function is defined as:

    .. math::

        erf(x) = \\frac{1}{\\sqrt{\\pi}} \\int_{-x}^{x} e^{t^2} dt
    """
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))


def opt_err_func(params, x, y, func):
    """
    Error function for fitting a function using non-linear optimization.

    Parameters
    ----------
    params : tuple
        A tuple with the parameters of `func` according to their order of
        input

    x : float array
        An independent variable.

    y : float array
        The dependent variable.

    func : function
        A function with inputs: `(x, *params)`

    Returns
    -------
    float array
        The marginals of the fit to x/y given the params
    """
    return y - func(x, *params)


class Dataset(object):
    """Class for generating carpet plot from fMRI data and fitting PCA to it"""
    def __init__(self, fmri_file, mask_file,
                 output_dir, TR=2.0):
        """ Initialize a Dataset object and import data.

        Parameters
        ----------
        fmri_file : str
            Path to 4-D (3-D + time) functional MRI data in NIFTI format.
        mask_file : str
            Path to 3-D cmask in NIFTI format (e.g. cortical mask).
            Must have same coordinate space and data matrix as :fmri:
        output_dir : str
            Path to folder where results will be saved.
            If it doesn't exist, it's created.
        TR : float
            fMRI repetition time in seconds
            Default: 2.0
        """
        # Read parameters
        self.fmri_file = fmri_file
        self.mask_file = mask_file
        self.TR = TR
        # Create output directory if it doesn't exist
        if not os.path.isdir(output_dir):
            try:
                os.mkdir(output_dir)
            except IOError:
                print("Could not create 'output_dir'")
        self.output_dir = output_dir

        print("\nInitialized Dataset object")
        print(f"\tfMRI file: {fmri_file}")
        print(f"\tMask file: {mask_file}")
        print(f"\tOutput directory: {output_dir}")
        print("\tTR: {0:.2f} seconds".format(TR))

        # Call initializing functions
        print(f"Reading data...")
        self.data, self.mask = self.import_data()

    def import_data(self):
        """ Load fMRI and cortex_mask data using nibabel.

        Returns
        -------
        data : array
            A 4-D array containing fMRI data
        mask :
            A 3-D array containing mask data
        """

        # Check if input files exist and try importing them with nibabel
        if os.path.isfile(self.fmri_file):
            try:
                fmri_nifti = nib.load(self.fmri_file)
            except IOError:
                print(f"Could not load {self.fmri_file} using nibabel.")
                print("Make sure it's a valid NIFTI file.")
        else:
            print(f"Could not find {self.fmri_file} file ")

        if os.path.isfile(self.mask_file):
            try:
                mask_nifti = nib.load(self.mask_file)
            except IOError:
                print(f"Could not load {self.mask_file} using nibabel.")
                print("Make sure it's a valid NIFTI file.")
        else:
            print(f"Could not find {self.mask_file} file ")

        # Ensure that data dimensions are correct
        data = fmri_nifti.get_fdata()
        mask = mask_nifti.get_fdata()
        print(f"fMRI data read: dimensions {data.shape}")
        print(f"Cortex mask read: dimensions {mask.shape}")
        if len(data.shape) != 4:
            raise ValueError('fMRI must be 4-dimensional!')
        if len(mask.shape) != 3:
            raise ValueError('cortex_mask must be 4-dimensional!')
        if data.shape[:3] != mask.shape:
            raise ValueError('fMRI and cortex_mask must be in the same space')

        # read and store data dimensions
        self.x, self.y, self.z, self.t = data.shape
        # read header and affine from cortex_mask
        # will be used for saving NIFTI maps later
        self.header = mask_nifti.header
        self.affine = mask_nifti.affine

        return data, mask

    def get_carpet(self, tSNR_thresh=15.0, reorder=True, save=True):
        """ Makes a carpet plot from fMRI data

        Parameters
        ----------
        tSNR_thresh: float
            Voxels with tSNR values below this threshold will not be used.
            To deactivate set to `None`
            Default: 15.0
        reorder: boolean
            Whether to reorder carpet voxels according to their (decreasing)
            correlation with the global (mean across voxesl) signal
            Default: True
        save: boolean
            Whether to save the carpet matrix in the output directory.
            Default: True

        Returns
        -------
        carpet : array
            A 2-D array (voxels x time).
            Contains normalized fMRI data from within a mask.
        """

        # compute fMRI data mean, std, and tSNR across time
        data_mean = self.data.mean(axis=-1, keepdims=True)
        data_std = self.data.std(axis=-1, keepdims=True)
        data_tsnr = data_mean / (data_std + EPSILON)

        # Mask fMRI data array with 'mask'
        # Also mask voxels below tSNR threshold (if given)
        mask = self.mask < 0.5
        mask_4d = np.repeat(mask[:, :, :, np.newaxis], self.t, axis=3)
        tsnr_mask_4d = np.zeros(mask_4d.shape, dtype=bool)
        if tSNR_thresh is not None:
            tsnr_mask = data_tsnr.squeeze() < tSNR_thresh
            tsnr_mask_4d = np.repeat(tsnr_mask[:, :, :, np.newaxis],
                                     self.t, axis=3)
        data_masked = np.ma.masked_where(mask_4d | tsnr_mask_4d, self.data)

        # Reshape data in 2-d (voxels x time)
        data_2d = data_masked.reshape((-1, self.t))
        print(f"fMRI data reshaped to voxels x time {data_2d.shape}")

        # Get indices for non-masked rows (voxels)
        indices_valid = np.where(np.any(~np.ma.getmask(data_2d), axis=1))[0]
        print(f"{len(indices_valid)} voxels retained after masking")
        # Keep only valid rows in carpet matrix
        carpet = data_2d[indices_valid, :]
        print(f"Carpet matrix created with shape {carpet.shape}")

        # Normalize carpet (z-score)
        carpet = zscore(carpet, axis=1)
        print(f"Carpet matrix normalized to zero-mean unit-variance")

        # Re-order carpet plot based on correlation with the global signal
        if reorder:
            gs = np.mean(carpet, axis=0)
            gs_corr = pearsonr_2d(carpet, gs.reshape((1, self.t))).flatten()
            sort_index = [int(i) for i in np.flip(np.argsort(gs_corr))]
            carpet = carpet[sort_index, :]
            print('Carpet reordered')

        # Save carpet to npy file
        if save:
            np.save(os.path.join(self.output_dir, 'carpet_reordered.npy'),
                    carpet.data)
            print("Carpet matrix saved as 'carpet.npy'")

        self.carpet = carpet
        return

    def fit_pca_and_correlate(self, ncomp=5, save_pca_scores=False):
        """ Fits PCA to carpet and correlates the first
        :ncomp: components with all voxel time-series.
        Saves PCs and explained variance ratio

        Parameters
        ----------
        carpet : array
            2-D carpet matrix (voxels x time)
        ncomp : int
            Number of PCA components to retain.
            These are correlated with all carpet voxels
            Default: 5
        save_pca_scores: boolean
            Whether to save the PCA scores (transformed carpet)
            in the output directory.
            Default: False
        """

        # Fit PCA
        model = PCA(whiten=True)
        pca_scores = model.fit_transform(self.carpet)
        pca_comps = model.components_
        self.expl_var = model.explained_variance_ratio_

        # Save results to npy files
        np.save(os.path.join(self.output_dir, 'pca_components_all.npy'),
                pca_comps)
        np.save(os.path.join(self.output_dir,
                'pca_explained_variance_all.npy'), self.expl_var)
        if save_pca_scores:
            np.save(os.path.join(self.output_dir, 'pca_scores_all.npy'),
                    pca_scores)
        # Pass first ncomp PCs to pandas dataframe and save as csv
        comp_names = ['PC' + str(i + 1) for i in range(ncomp)]
        PCs = pd.DataFrame(data=model.components_.T[:, :ncomp],
                           columns=comp_names)
        PCs.to_csv(os.path.join(self.output_dir,
                   f'pca_components_{ncomp}.csv'), index=False)
        self.PCs = PCs

        # Correlate first ncomp PCs with carpet matrix

        return


class Model(object):
    """Class for fitting cumulative Gaussian functions to data"""
    def __init__(self, func=cumgauss):
        """ Initialize a model object.

        Parameters
        ----------
        data : Pandas DataFrame
            Data from a subjective contrast judgement experiment

        func : callable, optional
            A function that relates x and y through a set of parameters.
            Default: :func:`cumgauss`
        """
        self.func = func

    def fit(self, x, y, initial=[0.5, 1]):
        """
        Fit a Model to data.

        Parameters
        ----------
        x : float or array
           The independent variable: contrast values presented in the
           experiment
        y : float or array
           The dependent variable

        Returns
        -------
        fit : :class:`Fit` instance
            A :class:`Fit` object that contains the parameters of the model.

        """
        params, _ = opt.leastsq(opt_err_func, initial,
                                args=(x, y, self.func))
        return Fit(self, params)


class Fit(object):
    """
    Class for representing a fit of a model to data
    """
    def __init__(self, model, params):
        """
        Initialize a :class:`Fit` object.

        Parameters
        ----------
        model : a :class:`Model` instance
            An object representing the model used

        params : array or list
            The parameters of the model evaluated for the data

        """
        self.model = model
        self.params = params

    def predict(self, x):
        """
        Predict values of the dependent variable based on values of the
        indpendent variable.

        Parameters
        ----------
        x : float or array
            Values of the independent variable. Can be values presented in
            the experiment. For out-of-sample prediction (e.g. in
            cross-validation), these can be values
            that were not presented in the experiment.

        Returns
        -------
        y : float or array
            Predicted values of the dependent variable, corresponding to
            values of the independent variable.
        """
        return self.model.func(x, *self.params)
