import os

from setuptools import setup, find_packages

NAME = "tfgp"
DESCRIPTION = "MLGPLVM implementation in TF"
URL = "https://github.com/samuelmurray/TF-GP"
EMAIL = "samuelmu@kth.se"
AUTHOR = "Samuel Murray"
REQUIRES_PYTHON = ">=3.6.0"
LICENSE = "GNU General Public License v3.0"
TEST_DIR = "tests"

SCIKIT_LEARN_VERSION = ">=0.20"
TENSORFLOW_VERSION = ">=1.12.0"
TENSORFLOW_PROBABILITY_VERSION = ">=0.5.0"

REQUIRED = [
    "numpy",
]

EXTRA_REQUIRED = {
    "examples": [
        "IPython",
        "jupyter",
        "matplotlib",
        "pods",
        "scikit-learn" + SCIKIT_LEARN_VERSION,
        "scipy",
        "seaborn",
    ],
    "tf": [
        "tensorflow" + TENSORFLOW_VERSION,
        "tensorflow-probability" + TENSORFLOW_PROBABILITY_VERSION,
    ],
    "tf_gpu": [
        "tensorflow-gpu" + TENSORFLOW_VERSION,
        "tensorflow-probability" + TENSORFLOW_PROBABILITY_VERSION,
    ],
    "test": [
        "scikit-learn" + SCIKIT_LEARN_VERSION,
        "scipy",
    ],
}

# Read version number
version_dummy = {}
with open(os.path.join(NAME, '__version__.py')) as f:
    exec(f.read(), version_dummy)
VERSION = version_dummy["__version__"]
del version_dummy

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=(TEST_DIR,)),
    test_suite=NAME + "." + TEST_DIR,
    install_requires=REQUIRED,
    extras_require=EXTRA_REQUIRED,
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
