import setuptools
import io
import subprocess

project_name = "jllm"  
version = "4.0.5"

def get_version(version):
    try:
        tag = subprocess.check_output(
            ['git', 'describe', '--exact-match', '--tags', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if re.match(r'^v?(\d+\.\d+\.\d+)$', tag):
            return tag.lstrip('v')
        else:
            commit = _get_commit_short()
            return f"{version}+{commit}"

    except subprocess.CalledProcessError:
        commit = _get_commit_short()
        return f"{version}+{commit}"
    except Exception:
        return version

def _get_commit_short():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit
    except Exception:
        return 'unknown'

version = get_version(version)

setuptools.setup(
    name=project_name,
    version=version,
    author="Jian Lu",
    license="Apache 2.0",
    description=("Running Large Language Model easily, faster and low-cost."),

    url="https://github.com/janelu9/EasyLLM",
    project_urls={
        "Homepage": "https://github.com/janelu9/EasyLLM",
    },
    long_description=io.open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.9', 
    install_requires=[
    "deepspeed",
    "protobuf",
    "sentencepiece",
    "transformers",
    "pyarrow",
    "tiktoken"
    ],
)
