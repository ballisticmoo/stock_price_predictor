from setuptools import setup, find_packages

setup(
    name='stock_price_predictor',
    version='0.1',
    author="CS Wizards",
    author_email="ballisticmoo143@gmail.com",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'requests',
    ],
    entry_points={
        'console_scripts': [
            'stock_predictor=stonks:main',  # Change 'your_module_name' to the actual name of your Python module
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
