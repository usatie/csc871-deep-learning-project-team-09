from setuptools import setup, find_packages

setup(
    name="transformer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.1",
        "torchtext>=0.15.2",
        "torchdata>=0.6.1",
        "spacy>=3.8.5",
        "spacy-pkuseg>=0.0.33",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "portalocker>=2.8.2",
        "altair>=5.2.0",
        "GPUtil>=1.4.0",
    ],
    python_requires=">=3.8",
) 