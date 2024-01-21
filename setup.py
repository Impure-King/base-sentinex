"""setup.py for Sentinex"""

import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except OSError:
  README = ""

__version__ = "0.0.1"

with open("sentinex/version.py") as f:
  exec(f.read(), globals())

install_requires = [
    "numpy>=1.12",
    "jaxtyping",
    "optax",
    "equinox",
    "jax-dataloader",
    "tensorflow-datasets",
    "termcolor",
    "keras-core",
]

tests_require = [
    "atari-py==0.2.5",  # Last version does not have the ROMs we test on pre-packaged
    "clu",  # All examples.
    "gym==0.18.3",
    "jaxlib",
    "jraph>=0.0.6dev0",
    "ml-collections",
    "mypy",
    "opencv-python",
    "pytest",
    "pytest-cov",
    "pytest-custom_exit_code",
    "pytest-xdist==1.34.0",  # upgrading to 2.0 broke tests, need to investigate
    "pytype",
    "sentencepiece",  # WMT example.
    "tensorflow_text>=2.4.0",  # WMT example.
    "tensorflow_datasets",
    "tensorflow",
    "torch",
]

setup(
    name="tensorwrap",
    version=__version__,
    description="Sentinex: A high level interface aimed towards rapid prototyping and intuitive workflow for JAX.",
    long_description="\n\n".join([README]),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Lelouch",
    author_email="ImpureK@gmail.com",
    url="https://github.com/Impure-King/base-sentinex",
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
        },
    )
