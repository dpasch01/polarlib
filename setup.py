import setuptools

with open("README.md", "r", encoding = "utf-8") as fh: long_description = fh.read()

setuptools.setup(
    name                          = "polarlib",
    version                       = "0.0.1",
    author                        = "Demetris Paschalides",
    author_email                  = "dpasch01@ucy.ac.cy",
    description                   = "N/A",
    long_description              = long_description,
    long_description_content_type = "text/markdown",
    package_dir                   = {"": "polarlib"},
    packages                      = setuptools.find_packages(where="polarlib"),
    python_requires               = ">=3.9"
)
