from setuptools import setup, find_packages
import os
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()
    print(f"Requirements loaded: {requirements}")

setup(
    name="expressivity_project",
    version="0.1.0",
    description="A tool to do calculations of expressivity in machine learning models",
    long_description=long_description,
    author="Mohammad Amirifard",
    author_email="xxx",
    url="xxx",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='==3.9.23',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)