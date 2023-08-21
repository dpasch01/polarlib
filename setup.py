import setuptools

with open("README.md", "r") as fh: long_description = fh.read()

with open('./requirements.txt') as f: reqs = f.read().strip().split('\n')

setuptools.setup(
    name                          = "polarlib",
    version                       = "0.0.1",
    author                        = "Demetris Paschalides",
    author_email                  = "dpasch01@ucy.ac.cy",
    description                   = "Python library for POLAR, a framework for the modelling of polarization and identification of polarizing topics in news articles.",
    long_description              = long_description,
    long_description_content_type = "text/markdown",
    packages                      = setuptools.find_packages(where="src"),
    package_dir                   = {'': 'src'},
    python_requires               = '>=3.9',
    install_requires              = reqs
)