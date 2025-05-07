from setuptools import setup, find_packages

setup(
    name="transformer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.7.0",
        "spacy>=3.8.5",
        "GPUtil>=1.4.0",
    ],
    python_requires=">=3.9",
)
