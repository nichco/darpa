from setuptools import setup, find_packages

setup(
    name='darpa',
    version='0',
    packages=[
        'darpa',
    ],
    install_requires=[
        'csdl',
    ],
    package_data={'': ['*.mat']}
)