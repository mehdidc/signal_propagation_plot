import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


requirements = [
    # use environment.yml
    "numpy"
]


setup(
    name="signal_propagation_plot",
    version="0.0.1",
    url="https://github.com/mehdidc/signal_propagation_plot",
    author="Mehdi Cherti",
    author_email="m.cherti@fz-juelich.de",
    description="Short description",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
    ],
)
