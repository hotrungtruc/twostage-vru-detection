"""
Setup script for DINOv3 package.

Usage:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="dinov3",
    version="1.0.0",
    description="DINOv3: A comprehensive package for DINOv3 model inference, classification, and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DINOv3 Computer Vision Team",
    author_email="contact@dinov3-traffic.ai",
    url="https://github.com/dinov3-traffic/dinov3",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="dinov3 vision transformers computer-vision object-detection tracking",
    include_package_data=True,
    zip_safe=False,
)
