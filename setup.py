from setuptools import setup, find_packages
import os

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "source", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]

with open("README.md", mode="r") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", mode="r") as f:
    REQUIREMENTS = f.read()

setup(
    name="hmp",
    version=VERSION,
    author="Jayanta Dey",
    author_email="jayanta.dey@utsa.edu",
    maintainer="Jayanta Dey",
    maintainer_email="jayanta.dey@utsa.edu",
    description="A a package for exploring and using sleep model built on temporal scaffolding hypothesis",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/jdey4/sleep_experiment/",
    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12.7"
    ],
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=["tests", "tests.*", "tests/*"]),
    include_package_data=True
)