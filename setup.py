import setuptools

with open("README.md", "r") as fh: long_description = fh.read()

with open('./requirements.txt') as f: reqs = f.read().strip().split('\n')

setuptools.setup(
    name                          = "POLARLib",
    version                       = "0.0.1",
    author                        = "Demetris Paschalides",
    author_email                  = "dpasch01@ucy.ac.cy",
    description                   = "N/A",
    long_description              = long_description,
    long_description_content_type = "text/markdown",
    packages                      = setuptools.find_packages(where="polarlib"),
    package_dir                   = {'': 'polarlib'},
    python_requires               = '>=3.9',
    install_requires              = reqs
)