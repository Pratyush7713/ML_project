from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)-> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Pratyush',
    author_email='pratyush9198@gmail.com',
    packages= find_packages(),  ## It will look for all the folders if that folder contains __init__.py file then this command will create that folder as a package.
    install_requires= get_requirements('requirements.txt')
)