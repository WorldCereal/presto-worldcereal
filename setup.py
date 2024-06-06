from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = "\n".join(
    [line for line in long_description.split("\n") if not line.startswith("<img")]
)

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
    install_requires=[
        "einops==0.6.0",
        "numpy==1.23.5",
        "torch==2.0",
        "tqdm==4.64.1",
        "rioxarray==0.13.1",
    ],
    python_requires=">=3.9",
    include_package_data=True,
)
