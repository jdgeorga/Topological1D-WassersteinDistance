from setuptools import setup, find_packages

setup(
    name="topological1d_wasserstein",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.1',
        'numpy',
        'scipy',
        'ase',
        'nequip',
        'moirecompare',
        'matplotlib',
        'psutil',
        'ot',
    ],
    author="Johnathan Dimitrios Georgaras",
    author_email="jdgeorga@stanford.edu",
    description="Analysis of structural changes in 2D materials using Wasserstein distances",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Topological1D-WassersteinDistance",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
        extras_require={
        'test': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'pytest-mock>=3.0',
        ],
    },
)