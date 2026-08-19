import os
from setuptools import setup, find_packages

reqs_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
requirements = []
if os.path.exists(reqs_path):
    with open(reqs_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="clip",
    py_modules=["clip"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements,
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
