import os

from setuptools import setup, find_packages

NAME = "tfgp"
DESCRIPTION = "MLGPLVM implementation in TF"
URL = "https://github.com/samuelmurray/TF-GP"
EMAIL = "samuelmu@kth.se"
AUTHOR = "Samuel Murray"
PYTHON_VERSION = ">=3.6.0"
LICENSE = "GNU General Public License v3.0"
TEST_DIR = "tests"

TENSORFLOW_VERSION = ">=1.12.0"
TENSORFLOW_PROBABILITY_VERSION = ">=0.5.0"

EXTRA_REQUIRED = {
    "oilflow": [
        "pods",
    ],
    "tf": [
        "tensorflow" + TENSORFLOW_VERSION,
        "tensorflow-probability" + TENSORFLOW_PROBABILITY_VERSION,
    ],
    "tf_gpu": [
        "tensorflow-gpu" + TENSORFLOW_VERSION,
        "tensorflow-probability" + TENSORFLOW_PROBABILITY_VERSION,
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
    python_requires=PYTHON_VERSION,
    url=URL,
    packages=find_packages(exclude=(TEST_DIR,)),
    test_suite=NAME + "." + TEST_DIR,
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
        "Programming Language :: Python :: 3.7",
    ],
)
