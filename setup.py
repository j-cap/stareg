#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="stareg", # the name of your package as it will appear on PyPI
    version="1.0.0",    # 0.0.x imply that it is unstable
    url="https://github.com/j-cap/stareg",
    author="Jakob Weber",
    author_email="jakobweber@hotmail.com",
    description="Implementation of Structured Additive Regression",  # one liner
    py_modules=[
        "bspline",
        "penalty_matrix",
        "smooth",
        "star_model",
        "tensorproductspline",
        "utils" 
        ], # list of actual python code modules -> this is what is imported
    packages=find_packages(exclude=("tests",)),
    license="MIT",
    classifiers=[ # to search for it on PyPI
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [ # describe the used libraries and versions, e.g. pandas, numpy == production 
                         # dependencies -> versions should be as relaxed as possible
        "numpy",
        "pandas",
        "sklearn",
        "plotly"
    ],
    extras_require = {  # for optional dependencies, e.g testing 
                        #  -> versions should be as specific as possible 
        "dev": [
            "pytest>=3.7",
        ],
    },
    python_requires=">=3.6",
)