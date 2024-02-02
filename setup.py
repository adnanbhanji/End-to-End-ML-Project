from setuptools import find_packages, setup

def get_requirements(file_path):
    '''
    this function will return list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', ' ') for req in requirements]

setup(
    name='end_to_end_project',
    version='0.0.1',
    author='abhanji',
    author_email='abhanji@student.ie.edu',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
)