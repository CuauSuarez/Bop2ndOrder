from os.path import join, dirname, abspath
from setuptools import setup, find_packages


def readme():
    with open("README.md", "r") as f:
        return f.read()


def install_requires():
    with open(join(dirname(abspath(__file__)), "requirements.txt"), "r") as f:
        return f.read().splitlines()


setup(
    name="bop-2nd-order",
    version="0.1.0",
    author="Cuau",
    author_email="cuaudsr@gmail.com",
    description='Implementation for paper "Bop and Beyond"',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/plumerai/rethinking-bnn-optimization.git",
    packages=find_packages(),
    python_requires=">=3.6",
    license="Apache 2.0",
    install_requires=install_requires(),
    extras_require={
        "tensorflow": ["tensorflow==2.0.0"],
        "tensorflow_gpu": ["tensorflow-gpu==2.0.0"],
    },
    entry_points="""
        [console_scripts]
        bnno=bnn_optimization.train:cli
    """,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
    ],
)
