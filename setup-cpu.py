from setuptools_rust import Binding, RustExtension
from setuptools import setup, find_packages


def make_rust_extension(module_name):
    return RustExtension(
        module_name, "Cargo.toml", debug=False, binding=Binding.PyO3
    )

setup(
    name="damavand-cpu",
    version="0.1.19",
    rust_extensions=[ make_rust_extension("damavand") ],
    packages=find_packages("./", exclude=("./tests",)),
    include_package_data=True,
    setup_requires=["setuptools", "setuptools-rust"],
    zip_safe=False,
)
