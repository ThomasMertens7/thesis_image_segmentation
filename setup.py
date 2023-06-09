from setuptools import setup, find_packages

setup(
    name='geomeansegmentation',
    version='0.1.1',
    description='A Python package for computing matrix geometric mean and performing image segmentation',
    author='Robbe Ramon',
    author_email='robbe.ramon@gmail.com',
    url='https://https://github.com/thesis-robbe-ramon-2023/matrix-geometric-mean-image-segmentation',
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'matrix geometric mean', 'image segmentation'],
    classifiers=[
        'Intended Audience :: Education',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)