#!/usr/bin/env python3
"""
HARPO: Hierarchical Agentic Reasoning with Preference Optimization
for Conversational Recommendation

Setup script for package installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="harpo",
    version="1.0.0",
    author="HARPO Authors",
    author_email="harpo@example.com",
    description="HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harpo-bench/harpo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "harpo=cli:cli",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "flash": [
            "flash-attn>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "harpo-train=scripts.train:main",
            "harpo-eval=scripts.evaluate:main",
            "harpo-chat=scripts.chat:main",
        ],
    },
)