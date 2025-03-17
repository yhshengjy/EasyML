from setuptools import setup, find_packages

setup(
    name='EasyML',
    version='1.0',
    packages=find_packages(),
    package_data={
        'easyml': ['datasets/*.csv'],
    }

)