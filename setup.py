from setuptools import setup, find_packages

setup(
    name='openmpc',  # Replace with your project name
    version='0.1.0',  # Version of your project
    author='Mikael Johansson',
    author_email='mikaelj@kth.se',
    description='OpenMPC is a simple, flexible, and extensible MPC toolbox tailored for education and research. It builds on cvxpy, casadi, cddlib and the Python control package.',
    long_description=open('README.md').read(),  # Optional: read from README file
    long_description_content_type='text/markdown',  # Optional: if you're using Markdown
    url='https://github.com/mikaelj-kth-se/OpenMPC',  # Optional: Project's URL
    packages=find_packages(),  # Automatically find all packages and sub-packages
    install_requires=[
        'numpy>=1.21.0',  # List your dependencies here
        'casadi>=3.6.3',
        'cvxpy[mosek]>=1.3.0',
        'pycddlib>=3.0.0b6',
        'matplotlib',
        'control'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the minimum Python version
)
