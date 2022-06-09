from setuptools import setup, find_packages

setup(
    name="AdrianPack",
    author="Adrian v Eik",
    version="0.0.2.2",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "matplotlib", "scipy"],
    tests_require=["pytest"]
)
