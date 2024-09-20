from setuptools import setup, find_packages

setup(
    name='frg-electrochemistry',           # Name of your library
    version='0.0.1',                    # Version of your library
    packages=find_packages(),            # Automatically find packages
    install_requires=[                   # List of dependencies
        'impedance',
        'scipy',
        'numpy',
        'matplotlib'
    ],
    author='Tejas Nivarty',
    author_email='tnivarty@ucsd.edu',
    description='Tools for analyzing electrochemistry data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Tejas-Nivarty/Electrochemistry',  # Your GitHub URL
    classifiers=[                        # Optional, metadata for package search
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11.9',             # Minimum Python version
)