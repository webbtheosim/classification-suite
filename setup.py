from setuptools import find_packages, setup

setup(
    name='ClassificationSuite',
    version='1.0.0',
    description='Data-efficient protocols for classification.',
    packages=find_packages('./src'),
    package_dir={'': 'src'}
)
