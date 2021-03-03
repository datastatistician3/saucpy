import codecs
import os.path, sys
import re

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


def fpath(name):
    return os.path.join(os.path.dirname(sys.argv[0]), name)

def read(fname):
    return codecs.open(fpath(fname), encoding='utf-8').read()

def grep(attrname):
    pattern = r"{0}\W*=\W*'([^']+)'".format(attrname)
    strval, = re.findall(pattern, file_text)
    return strval


file_text = read(fpath('saucpy/__init__.py'))

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='saucpy',
    version='0.0.1',
    description='Perform AUC analyses with discrete covariates and a semi-parametric estimation',
    long_description=read(fpath('README.md')),
    url='https://github.com/sbohora/saucpy/',
    author='Som B. Bohora',
    author_email="energeticsom@gmail.com",
    license='MIT',
    packages=find_packages(exclude = ['saucpy.examples', 'saucpy.tests']),
    include_package_data = True,
    zip_safe=False,
    install_requires=[
        'python-dateutil',
        'numpy',
        'pandas',
        'patsy',
        'scipy'
        ],
    test_suite="tests",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Text Processing :: Linguistic'
      ]
)