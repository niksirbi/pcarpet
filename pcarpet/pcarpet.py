from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, gaussian_kde
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Matoplotlib parameters for saving vector figures properly
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

__all__ = ["pearsonr_2d", "get_axis_coords", "Dataset"]

# Small value added to some denominators to avoid zero division
EPSILON = 1e-9


def pearsonr_2d(A, B):
    """Calculates row-wise Pearson's correlation between 2 2d-arrays

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

    # Check if the 2 input arrays are 2d and have the same column number T
    if (A.ndim != 2) or (B.ndim) != 2:
        raise ValueError('A and B must be 2d numpy arrays.')
    if A.shape[1] != B.shape[1]:
        raise ValueError('A and B arrays must have the same shape.')

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


def get_axis_coords(fig, ax):
    """Gets various coordinates of an axis
    within the figure space.

    Parameters
    ----------
    fig : matplotlib figure object
    ax : matplotlib axis object

    Returns
    -------
    coords : dictionary
        Contains the various coordinates:
        xmin, xmax, ymin, ymax, W (width), H (height),
        xcen (x center), ycen (ycenter)
    """
    try:
        size_x, size_y = fig.get_size_inches() * fig.dpi
    except ValueError:
        print('fig must be a matplotlib figure object')

    try:
        box = ax.bbox
    except ValueError:
        print('ax must be a matplotlib axis object')

    xmin, xmax = box.xmin, box.xmax
    ymin, ymax = box.ymin, box.ymax
    x0 = xmin / size_x
    x1 = xmax / size_x
    y0 = ymin / size_y
    y1 = ymax / size_y
    width = x1 - x0
    height = y1 - y0
    xc = x0 + width / 2
    yc = y0 + height / 2
    coords = {'xmin': x0, 'xmax': x1,
              'ymin': y0, 'ymax': y1,
              'W': width, 'H': height,
              'xcen': xc, 'ycen': yc}
    return coords


class Dataset(object):
    """Class for generating carpet plot from fMRI data and fitting PCA to it"""
    def __init__(self, fmri_file, mask_file, output_dir):
        """ Initialize a Dataset object and import data.

        Parameters
        ----------
        fmri_file : str
            Path to 4d (3d + time) functional MRI data in NIFTI format.
        mask_file : str
            Path to 3d mask in NIFTI format (e.g. cortical mask).
            Must have same coordinate space and data matrix as :fmri:
        output_dir : str
            Path to folder where results will be saved.
            If it doesn't exist, it's created.
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

    def import_data(self):
        """ Loads fMRI and mask data using nibabel.
        """

        print("Reading data...")
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

        # store data and mask variables as object attributes
        self.data = data
        self.mask = mask
        return

    def get_carpet(self, tSNR_thresh=15.0,
                   reorder_carpet=True, save_carpet=False):
        """ Makes a carpet matrix from fMRI data.
        A carpet is a 2d matrix shaped voxels x time which contains
        the normalized (z-score) BOLD-fMRI signal from within a mask

        Parameters
        ----------
        tSNR_thresh : float or None
            Voxels with tSNR values below this threshold will be excluded.
            To deactivate set to None.
            Default: 15.0
        reorder_carpet : boolean
            Whether to reorder carpet voxels according to their (decreasing)
            correlation with the global (mean across voxels) signal
            Default: True
        save_carpet : boolean
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
        print("Carpet normalized to zero-mean unit-variance.")

        # Re-order carpet plot based on correlation with the global signal
        if reorder_carpet:
            gs = np.mean(carpet, axis=0)
            gs_corr = pearsonr_2d(carpet, gs.reshape((1, self.t))).flatten()
            sort_index = [int(i) for i in np.flip(np.argsort(gs_corr))]
            carpet = carpet[sort_index, :]
            print('Carpet reordered.')

        # Save carpet to npy file
        if save_carpet:
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
        np.save(os.path.join(self.output_dir, 'PCs.npy'),
                self.pca_comps)
        np.save(os.path.join(self.output_dir,
                             'PCA_expl_var.npy'),
                self.expl_var)
        if save_pca_scores:
            np.save(os.path.join(self.output_dir, 'PCA_scores.npy'),
                    pca_scores)
        print("PCA fit to carpet and results saved.")
        return

    def correlate_with_carpet(self, ncomp=5, flip_sign=True):
        """ Correlates the first ncomp principal components (PCs)
        with all carpet voxel time-series. Saves the correlation matrix.

        Parameters
        ----------
        ncomp : int
            Number of PCA components to retain. These first PCs (fPCs)
            are correlated with all carpet voxels.
            Default: 5
        flip_sign : boolean
            If True, an fPC (and its correlation values) will be sign-flipped
            when the median of its original correlation with carpet voxels is
            negative. This enforces the sign of the fPC to match the sign of
            the BOLD signal activity for most voxels. The sign-flipped
            fPCs are only used for downstream analysis and visualization
            (the saved PCA components and scores retain the original sign).
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
        fPCs.to_csv(os.path.join(self.output_dir, 'fPCs.csv'), index=False)

        # Correlate fPCs with carpet matrix
        fPC_carpet_R = pearsonr_2d(self.carpet, fPCs.values.T)
        # Save correlation matrix (voxels x ncom) as npy
        np.save(os.path.join(self.output_dir, 'fPCs_carpet_corr.npy'),
                fPC_carpet_R)
        print(f"First {ncomp} PCs correlated with carpet.")

        # Construct table reporting various metrics for each fPC
        report = pd.DataFrame()
        report.loc[:, 'PC'] = comp_names
        report.loc[:, 'expl_var'] = self.expl_var[:ncomp]
        report.loc[:, 'carpet_r_median'] = [np.median(fPC_carpet_R[:, i])
                                            for i in range(self.ncomp)]
        report.loc[:, 'sign_flipped'] = [False] * self.ncomp

        # Flip sign if asked
        N_flipped = 0
        if flip_sign:
            for i, c in enumerate(fPCs):
                if report.loc[i, 'carpet_r_median'] < 0:
                    fPCs[c] = -1 * fPCs[c]
                    fPC_carpet_R[:, i] = -1 * fPC_carpet_R[:, i]
                    report.loc[i, 'sign_flipped'] = True
                    N_flipped += 1
        # If any flips occured, save flipped fPCs and their carpet correlation
        if N_flipped > 0:
            fPCs.to_csv(os.path.join(self.output_dir, 'fPCs_flipped.csv'),
                        index=False)
            np.save(os.path.join(self.output_dir,
                                 'fPCs_carpet_corr_flipped.npy'),
                    fPC_carpet_R)
            print(f"Out of these, {N_flipped} sign-flipped.")

        # Save report table
        report.to_csv(os.path.join(self.output_dir,
                                   'fPCs_carpet_corr_report.csv'),
                      index=False)

        self.fPCs = fPCs
        self.fPC_carpet_R = fPC_carpet_R
        return

    def correlate_with_fmri(self):
        """ Correlates the retained (and possibly sign-flipped)
        first ncomp PCs (fPCs) with the original 4d fMRI dataset
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
                                   'fPCs_fMRI_corr.nii.gz')
        nib.save(output_nifti, output_file)
        print(f"First {self.ncomp} PCs correlated with fMRI data.")
        return

    def plot_report(self, TR='auto'):
        """ Plots a report of the results, including the carpet plot,
        the first ncomp PCs (fPCs), their correlation with the carpet,
        and their explained variance ratios. The plot image is saved
        in '.png' (raster) and '.svg' (vector) formats.

        Parameters
        ----------
        TR : 'auto' or float
            fMRI repetition time in seconds. If 'auto', the program
            attempts to read TR from the fMRI header. This can be
            bypassed by explicitly passing TR as a float.
            Default: 'auto'
        """

        if type(TR) in [int, float]:
            self.TR = float(TR)
            print(f"TR set to {self.TR:.3f} seconds")
        elif TR == 'auto':
            self.TR = float(self.header['pixdim'][4])
            print(f"TR of {self.TR:.3f} seconds read from fMRI header")
        else:
            raise ValueError("TR must be a float or 'auto'")

        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(left=0.05, right=0.95, hspace=0.1,
                            bottom=0.05, top=0.92, wspace=0.5)
        npc = self.ncomp

        # Carpet plot
        ax1 = plt.subplot2grid((6 + npc, 5), (0, 0),
                               rowspan=5, colspan=3)
        carpet_plot = ax1.imshow(self.carpet, interpolation='none',
                                 aspect='auto', cmap='Greys_r',
                                 vmin=-2, vmax=2)
        ax1.set_xlabel(f'Time ({self.carpet.shape[1]} TRs)')
        ax1.set_ylabel(f'Space ({self.carpet.shape[0]} voxels)')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Carpet plot: normalized BOLD (z-score)')
        # Plot carpet colorbar in a separate axis
        ax1_coords = get_axis_coords(fig, ax1)
        cb_ax1 = plt.axes([ax1_coords['xcen'] - 0.05,
                           ax1_coords['ymax'] + 0.06, 0.1, 0.01])
        plt.colorbar(carpet_plot, cax=cb_ax1, ticks=[-2, 0, 2],
                     orientation='horizontal')

        # Plot fPCs
        ymin = np.min(self.fPCs.values)
        ymax = np.max(self.fPCs.values)
        for i in range(npc):
            axpc = plt.subplot2grid((6 + npc, 5), (6 + i, 0), colspan=3)
            pc = self.fPCs[self.fPCs.columns[i]]
            axpc.plot(pc, color='0.2', lw=1.5)
            axpc.set_ylim(ymin, ymax)
            axpc.set_xlim(0, self.t)
            axpc.axis('off')
            axpc_coords = get_axis_coords(fig, axpc)
            fig.text(axpc_coords['xmin'] - 0.015, axpc_coords['ycen'],
                     self.fPCs.columns[i], ha='right', va='center')
            if i == 0:
                axpc.set_title('Principal Components (PCs)')
        # Plot time scalebar
        tax = plt.subplot2grid((6 + npc, 5), (5, 0), rowspan=1, colspan=3)
        tleft = self.t - 120 / self.TR
        tright = self.t - 60 / self.TR
        tcenter = self.t - 90 / self.TR
        tax.plot([tleft, tright], [0.8, 0.8], lw=1.5, color='k')
        tax.plot([tleft, tleft], [0.72, 0.88], lw=1.5, color='k')
        tax.plot([tright, tright], [0.72, 0.88], lw=1.5, color='k')
        tax.text(tcenter, 0.6, '1 minute', ha='center', va='top', color='k')
        tax.set_xlim(0, self.t),
        tax.set_ylim(0, 1)
        tax.axis('off')
        tax.patch.set_alpha(0.0)

        # Plot fPC-carpet correlations as matrix
        ax3 = plt.subplot2grid((6 + npc, 5), (0, 3), rowspan=5, colspan=1)
        R_matrix = ax3.imshow(self.fPC_carpet_R, interpolation='none',
                              aspect='auto', cmap='coolwarm',
                              vmin=-1, vmax=1)
        ax3.set_xticks(np.arange(npc))
        ax3.set_xticklabels(np.arange(1, npc + 1))
        ax3.set_yticks([])
        ax3.set_xlabel('PCs')
        ax3.set_ylabel(f'Space ({self.carpet.shape[0]} voxels)')
        ax3.tick_params(axis='both', length=0)
        ax3.set_title('Correlation (r)')
        # Plot correlation colorbar in a separate axis
        ax3_coords = get_axis_coords(fig, ax3)
        cb_ax3 = plt.axes([ax3_coords['xcen'] - 0.05,
                           ax3_coords['ymax'] + 0.06, 0.1, 0.01])
        plt.colorbar(R_matrix, cax=cb_ax3, ticks=[-1, 0, 1],
                     orientation='horizontal')

        # Plot PC-carpet correlation (r) histograms
        for i in range(npc):
            axh = plt.subplot2grid((6 + npc, 5), (6 + i, 3),
                                   rowspan=1, colspan=1)
            if i == 0:
                axh.set_title('Histograms')
            n, bins, patches = axh.hist(self.fPC_carpet_R[:, i], bins=50,
                                        density=True, edgecolor=None,
                                        range=(-1, 1), lw=0)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            norm = Normalize(vmin=-1, vmax=1)
            pcolors = [ScalarMappable(norm=norm, cmap='coolwarm').to_rgba(b)
                       for b in bin_centers]
            for p, c in zip(patches, pcolors):
                plt.setp(p, 'facecolor', c)
            kde = gaussian_kde(self.fPC_carpet_R[:, i])
            kde_samples = np.linspace(-1, 1, 100)
            axh.plot(kde_samples, kde(kde_samples), lw=1.5, color='0.2')
            axh.set_xlim(-1, 1)
            axh.set_yticks([])
            axh.set_xticks([-1, 0, 1])
            axh.set_xlabel('Correlation (r)')
            if i == npc - 1:
                axh.set_xticklabels([-1, 0, 1])
            else:
                axh.set_xticklabels([])
            axh.spines['right'].set_visible(False)
            axh.spines['left'].set_visible(False)
            axh.spines['top'].set_visible(False)

        # Plot variance explained
        axv = plt.subplot2grid((6 + npc, 5), (0, 4), rowspan=5, colspan=1)
        axv.bar(np.arange(npc), 100 * self.expl_var[:npc],
                color='0.7', edgecolor='0.2', linewidth=1)
        axv.set_xticks(np.arange(npc))
        axv.set_xticklabels(np.arange(1, npc + 1))
        axv.set_xlabel('PCs')
        axv.set_title('Expl. variance (%)')

        # Plot median correlation
        medians = np.median(self.fPC_carpet_R, axis=0)
        axm = plt.subplot2grid((6 + npc, 5), (6, 4), rowspan=npc, colspan=1)
        bar_colors = [ScalarMappable(norm=norm, cmap='coolwarm').to_rgba(m)
                      for m in medians]
        axm.bar(np.arange(npc), medians, color=bar_colors,
                edgecolor='0.2', linewidth=1)
        axm.set_xticks(np.arange(npc))
        axm.set_xticklabels(np.arange(1, npc + 1))
        axm.set_xlabel('PCs')
        axm.set_title('Median correlation (r)')

        # Save figure
        plotname = 'fPCs_carpet_corr_report'
        plt.savefig(os.path.join(self.output_dir,
                                 f'{plotname}.png'), dpi=128)
        plt.savefig(os.path.join(self.output_dir,
                                 f'{plotname}.svg'))
        print(f"Visual report generated and saved as {plotname}.")

    def run_pcarpet(self, **kwargs):
        """ Runs the entire pcarpet pipeline using the default options
        for each function. The defaults can be overriden by passing
        the following optional keywords arguments:

        Parameters
        ----------
        tSNR_thresh : float or None
            Voxels with tSNR values below this threshold will be excluded
            from the carpet. To deactivate set to None.
            Default: 15.0
        reorder_carpet : boolean
            Whether to reorder carpet voxels according to their (decreasing)
            correlation with the global (mean across voxels) signal
            Default: True
        save_carpet : boolean
            Whether to save the carpet matrix in the output directory.
            The file might be large (possibly > 100MB depending on
            fMRI data and mask size).
            Default: False
        save_pca_scores : boolean
            Whether to save the PCA scores (transformed carpet)
            in the output directory. The file might be large
            (possibly > 100MB depending on fMRI data and mask size).
            Default: False
        ncomp : int
            Number of PCA components to retain. These first PCs (fPCs)
            are correlated with all carpet voxels and subsequently
            also with the entire fMRI dataset.
            Default: 5
        flip_sign : boolean
            If True, an fPC (and its correlation values) will be sign-flipped
            when the median of its original correlation with carpet voxels is
            negative. This enforces the sign of the fPC to match the sign of
            the BOLD signal activity for most voxels. The sign-flipped
            fPCs are only used for downstream analysis and visualization
            (the saved PCA components and scores retain the original sign).
            Default: True
        TR : 'auto' or float
            fMRI repetition time in seconds. If 'auto', the program
            attempts to read the TR from the fMRI header. This can be
            bypassed by explicitly passing TR as a float.
            Default: 'auto'
        """

        # Define default options in a dictionary
        options = {'tSNR_thresh': 15.0, 'reorder_carpet': True,
                   'save_carpet': False, 'save_pca_scores': False,
                   'ncomp': 5, 'flip_sign': True, 'TR': 'auto'}

        # Override default if any of the options is given
        # explicitly as a keyword argument
        for key, value in kwargs.items():
            if key in options.keys():
                options[key] = value
            else:
                print(f"'{key}' is not a valid argument!")
        self.used_options = options

        self.import_data()
        self.get_carpet(tSNR_thresh=options['tSNR_thresh'],
                        reorder_carpet=options['reorder_carpet'],
                        save_carpet=options['save_carpet'])
        self.fit_pca2carpet(save_pca_scores=options['save_pca_scores'])
        self.correlate_with_carpet(ncomp=options['ncomp'],
                                   flip_sign=options['flip_sign'])
        self.correlate_with_fmri()
        self.plot_report(TR=options['TR'])
