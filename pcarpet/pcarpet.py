from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
from sklearn.decomposition import PCA

__all__ = ["pearsonr_2d", "Dataset"]

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


class Dataset(object):
    """Class for generating carpet plot from fMRI data and fitting PCA to it"""
    def __init__(self, fmri_file, mask_file,
                 output_dir, TR='auto'):
        """ Initialize a Dataset object and import data.

        Parameters
        ----------
        fmri_file : str
            Path to 4d (3d + time) functional MRI data in NIFTI format.
        mask_file : str
            Path to 3d cmask in NIFTI format (e.g. cortical mask).
            Must have same coordinate space and data matrix as :fmri:
        output_dir : str
            Path to folder where results will be saved.
            If it doesn't exist, it's created.
        TR : 'auto' or float
            fMRI repetition time in seconds. If 'auto', the program
            attempts to read TR from the fMRI header. This can be
            bypassed by explicitly passing TR as a float.
            Default: 'auto'
        """

        self.fmri_file = fmri_file
        self.mask_file = mask_file
        # Create output directory if it doesn't exist
        if not os.path.isdir(output_dir):
            try:
                os.mkdir(output_dir)
            except IOError:
                print("Could not create 'output_dir'")
        self.output_dir = output_dir

        print("\nInitialized Dataset object:")
        print(f"\tfMRI file: {fmri_file}")
        print(f"\tMask file: {mask_file}")
        print(f"\tOutput directory: {output_dir}")
        if type(TR) in [int, float]:
            self.TR = float(TR)
            print(f"\tTR set to {self.TR:.3f} seconds")
        elif TR == 'auto':
            self.TR = TR
            pass
        else:
            raise ValueError("TR must be a float or 'auto'")

        # Call initializing functions
        print(f"Reading data...")
        self.data, self.mask = self.import_data()

    def import_data(self):
        """ Load fMRI and cortex_mask data using nibabel.

        Returns
        -------
        data : array
            A 4d array containing fMRI data
        mask :
            A 3d array containing mask data
        """

        # Check if input files exist and try importing them with nibabel
        if os.path.isfile(self.fmri_file):
            try:
                fmri_nifti = nib.load(self.fmri_file)
            except IOError:
                print(f"Could not load {self.fmri_file} using nibabel.")
                print("Make sure it's a valid NIFTI file.")
        else:
            print(f"Could not find {self.fmri_file} file.")

        if os.path.isfile(self.mask_file):
            try:
                mask_nifti = nib.load(self.mask_file)
            except IOError:
                print(f"Could not load {self.mask_file} using nibabel.")
                print("Make sure it's a valid NIFTI file.")
        else:
            print(f"Could not find {self.mask_file} file.")

        # Ensure that data dimensions are correct
        data = fmri_nifti.get_fdata()
        mask = mask_nifti.get_fdata()
        print(f"\tfMRI data read: dimensions {data.shape}")
        print(f"\tMask read: dimensions {mask.shape}")
        if len(data.shape) != 4:
            raise ValueError('fMRI must be 4-dimensional!')
        if len(mask.shape) != 3:
            raise ValueError('Mask must be 3-dimensional!')
        if data.shape[:3] != mask.shape:
            raise ValueError('fMRI and mask must be in the same space!')

        # read data dimensions, header, and affine
        self.x, self.y, self.z, self.t = data.shape
        self.header = fmri_nifti.header
        self.affine = fmri_nifti.affine
        # read TR from the fMRI header (if 'auto')
        if self.TR == 'auto':
            self.TR = float(self.header['pixdim'][4])
            print(f"\tTR of {self.TR:.3f} seconds read from fMRI header")
        return data, mask

    def get_carpet(self, tSNR_thresh=15.0, reorder=True, save=False):
        """ Makes a carpet matrix from fMRI data.
        A carpet is a 2d matrix shaped voxels x time which contains
        the normalized (z-score) BOLD-fMRI signal from within a mask

        Parameters
        ----------
        tSNR_thresh : float
            Voxels with tSNR values below this threshold will be excluded.
            To deactivate set to `None`
            Default: 15.0
        reorder : boolean
            Whether to reorder carpet voxels according to their (decreasing)
            correlation with the global (mean across voxels) signal
            Default: True
        save : boolean
            Whether to save the carpet matrix in the output directory.
            The file might be large (possibly > 100MB depending on
            fMRI data and mask size).
            Default: False
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
        print(f"fMRI data reshaped to voxels x time {data_2d.shape}.")

        # Get indices for non-masked rows (voxels)
        indices_valid = np.where(np.any(~np.ma.getmask(data_2d), axis=1))[0]
        print(f"{len(indices_valid)} voxels retained after masking.")
        # Keep only valid rows in carpet matrix
        carpet = data_2d[indices_valid, :].data
        print(f"Carpet matrix created with shape {carpet.shape}.")

        # Normalize carpet (z-score)
        carpet = zscore(carpet, axis=1)
        print(f"Carpet normalized to zero-mean unit-variance.")

        # Re-order carpet plot based on correlation with the global signal
        if reorder:
            gs = np.mean(carpet, axis=0)
            gs_corr = pearsonr_2d(carpet, gs.reshape((1, self.t))).flatten()
            sort_index = [int(i) for i in np.flip(np.argsort(gs_corr))]
            carpet = carpet[sort_index, :]
            print('Carpet reordered.')

        # Save carpet to npy file
        if save:
            np.save(os.path.join(self.output_dir, 'carpet.npy'), carpet)
            print("Carpet saved as 'carpet.npy'.")

        self.carpet = carpet
        return

    def fit_pca2carpet(self, save_pca_scores=False):
        """ Fits PCA to carpet matrix and saves the principal
        componens (PCs), the explained variance ratios,
        and optionally the PCA scores (PCA-tranformed carpet)

        Parameters
        ----------
        save_pca_scores : boolean
            Whether to save the PCA scores (transformed carpet)
            in the output directory. The file might be large
            (possibly > 100MB depending on fMRI data and mask size).
            Default: False
        """

        # Fit PCA
        model = PCA(whiten=True)
        pca_scores = model.fit_transform(self.carpet)
        self.pca_comps = model.components_
        self.expl_var = model.explained_variance_ratio_

        # Save results to npy files
        np.save(os.path.join(self.output_dir, 'PCs_all.npy'),
                self.pca_comps)
        np.save(os.path.join(self.output_dir,
                             'PCs_all_expl_variance_ratio.npy'),
                self.expl_var)
        if save_pca_scores:
            np.save(os.path.join(self.output_dir, 'PCA_scores_all.npy'),
                    pca_scores)
        print("PCA fit to carpet and results saved.")
        return

    def correlate_with_carpet(self, ncomp=5, flip_sign=True):
        """ Correlates the first :ncomp: principal components (PCs)
        with all carpet voxel time-series. Saves the correlation matrix.

        Parameters
        ----------
        ncomp : int
            Number of PCA components to retain. These first PCs (fPCs)
            are correlated with all carpet voxels.
            Default: 5
        flip_sign : boolean
            If True, a PC (and its correlation with carpet voxels)
            will be sign-flipped when the median of its original
            correlation with carpet voxels is negative.
            This enforces the sign of the PC to match the sign of
            the BOLD signal activity for most voxels. This applies
            only fPCs.The sign-flipped PCs are only used for downstream
            analysis and visualization (the saved PCA components, scores,
            and report have the original sign).
            Default: True
        """

        # Assert that ncomp can be taken as integer
        try:
            self.ncomp = int(ncomp)
        except ValueError:
            print("'ncomp' must be an integer!")

        # Pass first ncomp PCs (fPCs) to pandas dataframe and save as csv
        comp_names = ['PC' + str(i + 1) for i in range(self.ncomp)]
        fPCs = pd.DataFrame(data=self.pca_comps.T[:, :self.ncomp],
                            columns=comp_names)
        fPCs.to_csv(os.path.join(self.output_dir,
                                 f'First{self.ncomp}_PCs.csv'), index=False)

        # Correlate fPCs with carpet matrix
        fPC_carpet_R = pearsonr_2d(self.carpet, fPCs.values.T)
        np.save(os.path.join(self.output_dir,
                             f'First{self.ncomp}_PCs_carpet_corr.npy'),
                fPC_carpet_R)
        # Save correlation matrix (voxels x ncom) as npy
        print(f"First {ncomp} PCs correlated with carpet.")

        # Construct table reporting various metrics for each fPC
        report = pd.DataFrame()
        report.loc[:, 'PC'] = comp_names
        report.loc[:, 'expl_var'] = self.expl_var[:ncomp]
        report.loc[:, 'carpet_R_median'] = [np.median(fPC_carpet_R[:, i])
                                            for i in range(self.ncomp)]
        report.loc[:, 'sign_flipped'] = [False] * self.ncomp

        # Flip sign if asked
        N_flipped = 0
        if flip_sign:
            for i, c in enumerate(fPCs):
                if report.loc[i, 'carpet_R_median'] < 0:
                    fPCs[c] = -1 * fPCs[c]
                    fPC_carpet_R[:, i] = -1 * fPC_carpet_R[:, i]
                    report.loc[i, 'sign_flipped'] = True
                    N_flipped += 1
        # If any flips occured, save flipped fPCs and their carpet correlation
        if N_flipped > 0:
            fPCs.to_csv(os.path.join(self.output_dir,
                                     f'First{self.ncomp}_PCs_flipped.csv'),
                        index=False)
            np.save(os.path.join(self.output_dir,
                    f'First{self.ncomp}_PCs_flipped_carpet_corr.npy'),
                    fPC_carpet_R)
            print(f"Out of these, {N_flipped} sign-flipped.")

        # Save report table
        report.to_csv(os.path.join(self.output_dir,
                      f'First{self.ncomp}_PCs_carpet_corr_report.csv'),
                      index=False)

        self.fPCs = fPCs
        self.PC_carpet_R = fPC_carpet_R
        return

    def correlate_with_fmri(self):
        """ Correlates the retained (and possibly sign-flipped)
        first :ncomp: PCs (fPCs) with the original 4d fMRI dataset
        and saves the resulting correlation maps as a 4d NIFTI file
        (3d space + ncomp).
        """

        # Reshape 4d fMRI data into a 2d (voxels * time) matrix
        fmri_2d = self.data.reshape((-1, self.t))
        # Correlate with PCs
        fPC_fmri_R = pearsonr_2d(fmri_2d, self.fPCs.values.T)
        # Reshape correlation to 4d (3d space * components)
        fPC_fmri_R = fPC_fmri_R.reshape((self.x, self.y, self.z, self.ncomp))
        # Create appropriate NIFTI header
        header = self.header.copy()
        header['dim'][4] = self.ncomp
        header['pixdim'][4] = 1
        # Save correlation maps as NIFTI
        output_nifti = nib.Nifti1Image(fPC_fmri_R, self.affine,
                                       header=header)
        output_file = os.path.join(self.output_dir,
                                   f'First{self.ncomp}_PCs_fMRI_corr.nii.gz')
        nib.save(output_nifti, output_file)
        print(f"First {self.ncomp} PCs correlated with fMRI data.")
        return
