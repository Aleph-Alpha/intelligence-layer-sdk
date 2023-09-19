from setuptools import setup, find_packages  # type: ignore
from pathlib import Path

reqs_dir = Path("./requirements")


def read_requirements(filename: str):
    requirements_file = reqs_dir / filename
    if requirements_file.is_file():
        requirements_list = requirements_file.read_text().splitlines()
        return requirements_list
    else:
        return []


requirements_base = read_requirements("base.txt")
requirements_test = read_requirements("test.txt")


setup(
    name="intelligence_layer",
    url="https://gitlab.aleph-alpha.de/product/intelligence-layer",
    author="Samuel Weinbach",
    author_email="samuel.weinbach@aleph-alpha.com",
    install_requires=requirements_base,
    tests_require=requirements_test,
    extras_require={
        "test": requirements_test,
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.1.0",
    license="Aleph Alpha Licensed",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    entry_points="""""",
    package_data={
        # If any package contains *.json or *.typed
        "": ["*.json", "*.typed"],
    },
    include_package_data=True,
)
