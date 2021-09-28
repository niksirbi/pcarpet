Welcome to pcarpet's documentation!
====================================

`pcarpet` allows you to create a **carpet plot** from fMRI data and decompose it with **PCA**.

A 'carpet plot' here refers to a 2d representation of fMRI data (voxels x time), 
very similar to 'The Plot' described by Jonathan D Power (`Power 2017 
<https://www.sciencedirect.com/science/article/abs/pii/S1053811916303871?via%3Dihub>`_). 
It has been referred to with various other names, including 'grayplot'. 
This visual representation of fMRI data is suited for identifying wide-spread signal fluctutations 
(`Aquino et al. 2020 <https://www.sciencedirect.com/science/article/pii/S1053811920301014>`_), 
which often come from non-neural sources (e.g. head motion).

That said, the carpet plot can also reveal 'real' neural activity, especially when the activity is
very slow and synchronous, as is the case for anesthesia-induced burst-suppression (Sirmpiltze et al. 2021).
The `pcarpet` package implements the analytical pipeline used in the Sirmpiltze et al. 2021 paper.
The pipeline consists of the following steps:

To see how to use it, please refer to the `README file 
<https://github.com/niksirbi/pcarpet/blob/master/README.md>`_ in the Github repository.


Contents:

.. toctree::
   :maxdepth: 2

   example_usage.ipynb
   api
