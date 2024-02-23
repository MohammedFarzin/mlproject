from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path) -> List[str]:
    requirements = []
    with  open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "")for req in requirements]

        if '-e .' in requirements:
            requirements = requirements.remove('-e .')

    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Farzin',
    author_email='mohammedfarzin.m123@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)