from setuptools import setup, find_packages

# Be sure to update the version number in both "setup.py" and "meta.yaml" files, and the README and CITATION versions.
setup(
    name='tensorflow-engram',
    version='0.1.0',
    packages=find_packages(),
    author='Daniel Szelogowski',
    description='A Python package for Engram Neural Networks, adding biologically-inspired Hebbian memory and engram layers to TensorFlow/Keras models, supporting memory traces, plasticity, attention, and sparsity for neural sequence learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://danielathome19.github.io/Engram-Neural-Network',
    project_urls={
        'Documentation': 'https://danielathome19.github.io/Engram-Neural-Network',
        'Source': 'https://github.com/danielathome19/Engram-Neural-Network',
        'Tracker': 'https://github.com/danielathome19/Engram-Neural-Network/issues',
    },
    python_requires='>=3.12',
)
