from setuptools import setup, find_namespace_packages

setup(
    name="AdrianPack",
    author="Adrian v Eik",
    version="0.0.1",
    packages=find_namespace_packages(),
    install_requires=["numpy", "pandas", "matplotlib"],
    tests_require=["pytest"]
)
