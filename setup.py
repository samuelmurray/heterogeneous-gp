import os
from typing import Dict

from setuptools import setup, find_packages

NAME = "hgp"
DESCRIPTION = "MLGPLVM implementation in TF"
URL = "https://github.com/samuelmurray/heterogeneous-gp"
EMAIL = "samuelmu@kth.se"
AUTHOR = "Samuel Murray"
PYTHON_VERSION = ">=3.6.0"
LICENSE = "GNU General Public License v3.0"
TEST_DIR = "tests"

# Read version number
version_dummy: Dict[str, str] = {}
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
