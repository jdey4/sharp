from setuptools import setup, find_packages
import os

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

VERSION = None
with open(os.path.join(PROJECT_PATH, "sharp", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__ = "):
            VERSION = line.strip().split("=")[1].strip().strip('"').strip("'")
            break

if VERSION is None:
    raise RuntimeError("Could not find __version__ in source/__init__.py")

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    REQUIREMENTS = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="SHARP",
    version=VERSION,
    author="Jayanta Dey",
    author_email="jayanta.dey@utsa.edu",
    maintainer="Jayanta Dey",
    maintainer_email="jayanta.dey@utsa.edu",
    description="A package for exploring and using sleep models built on the temporal scaffolding hypothesis",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/jdey4/sleep_experiment/",
    license="Apache",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=["tests", "tests.*", "tests/*"]),
    include_package_data=True,
)