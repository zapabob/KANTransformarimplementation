from setuptools import setup, find_packages

setup(
    name="kan-transformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mosaicml-composer>=0.16.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "captum>=0.6.0",
        "optuna>=3.3.0",
        "wandb>=0.15.0",
        "plotly>=5.15.0",
        "dash>=2.11.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="神経調節機能付き三値活性化ネットワーク（KANモデル）",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kan-transformer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
) 