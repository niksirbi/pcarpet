from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 2
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = ''
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python :: 3",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "pcarpet: create a carpet plot from fMRI data and decompose it with PCA."
# Long description will go up on the pypi page
long_description = """

pcarpet
========
`pcarpet` is a small python package that creates a **carpet plot** from fMRI data and decomposes it with **PCA**.

**Citation:** Sirmpilatze et al. eLife 2022;11:e74813. DOI: https://doi.org/10.7554/eLife.74813

For an overview of the project, please refer to the `README file 
<https://github.com/niksirbi/pcarpet/blob/master/README.md>`_ in the Github repository 
and to the `documentation <https://pcarpet.readthedocs.io/en/latest/>`_.

License
=======
``pcarpet`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2021--, Nikoloz Sirmpilatze, German Primate Center.
"""

NAME = "pcarpet"
MAINTAINER = "Nikoloz Sirmpilatze"
MAINTAINER_EMAIL = "niko.sirbiladze@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/niksirbi/pcarpet"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Nikoloz Sirmpilatze"
AUTHOR_EMAIL = "niko.sirbiladze@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'pcarpet': [pjoin('data', '*')]}
REQUIRES = ['numpy', 'scipy', 'matplotlib', 'pandas',
            'scikit-learn', 'nibabel']
PYTHON_REQUIRES = ">= 3.7"
