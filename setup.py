from pathlib import Path
from typing import List

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = "\n".join(
    [line for line in long_description.split("\n") if not line.startswith("<img")]
)


def load_dependencies(tag: str) -> List[str]:
    """Load the dependencies from the specified requirements file."""
    with open(Path(__file__).parent / f"requirements.{tag}.txt") as f:
        requirements = [
            line.strip() for line in f.readlines() if line.strip() and (line[0] != "#")
        ]
    return requirements


# List all possible dependencies
run_dependencies = load_dependencies("inference")
train_dependencies = load_dependencies("training")
dev_dependencies = load_dependencies("dev")


setup(
    name="presto-worldcereal",
    description="Pretrained Remote Sensing Transformer (Presto) for WorldCereal",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gabriel Tseng",
    author_email="gabrieltseng95@gmail.com",
    version="0.0.1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    packages=["presto"] + [f"presto.{f}" for f in find_packages("presto")],
    install_requires=run_dependencies,
    python_requires=">=3.8",
    include_package_data=True,
    extras_require={
        "train": train_dependencies,
        "dev": dev_dependencies + train_dependencies,
    },
)
