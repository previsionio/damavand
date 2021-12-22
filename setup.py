from skbuild import setup
from setuptools_rust import Binding, RustExtension


def make_rust_extension(module_name):
    return RustExtension(
        module_name, "Cargo.toml", debug=False, features=["gpu"], binding=Binding.PyO3
    )

setup(
    name="damavand",
    version="0.1.0",
    rust_extensions=[ make_rust_extension("damavand") ],
    packages=["damavand"],
    include_package_data=True,
    zip_safe=False,
)
