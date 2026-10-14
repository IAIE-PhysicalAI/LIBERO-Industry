# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "./README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="libero",
    packages=find_packages(include=["libero", "libero.*"]),
    package_data={
        "libero.configs": ["**/*.yaml"],
        "libero.libero": [
            "assets/**/*",
            "bddl_files/**/*",
            "benchmark/*.json",
            "init_files/**/*",
        ],
    },
    # Runtime-specific dependencies are pinned in requirements-industry.txt and
    # requirements-deploy.txt because the two profiles use different simulators.
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.8,<3.11",
    description=(
        "LIBERO-Industry: Benchmarking Robotic Manipulation under Industrial "
        "Process Constraints, built on LIBERO"
    ),
    author="LIBERO Authors and LIBERO-Industry Authors",
    url="https://github.com/IAIE-PhysicalAI/LIBERO-Industry",
    project_urls={
        "Project Page": "https://IAIE-PhysicalAI.github.io/LIBERO-Industry/",
        "Source": "https://github.com/IAIE-PhysicalAI/LIBERO-Industry",
        "Upstream LIBERO": "https://github.com/Lifelong-Robot-Learning/LIBERO",
        "Issues": "https://github.com/IAIE-PhysicalAI/LIBERO-Industry/issues",
    },
    license="MIT",
    license_files=("LICENSE", "NOTICE"),
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "lifelong.main=libero.lifelong.main:main",
            "lifelong.eval=libero.lifelong.evaluate:main",
            "libero.config_copy=scripts.config_copy:main",
            "libero.create_template=scripts.create_template:main",
        ]
    },
)
