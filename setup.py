"""
Setup script for Deepfake Voice Detection System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deepfake-voice-detector",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered system to detect deepfake and synthetic voices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deepfake-voice-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepfake-detector-train=train:main",
            "deepfake-detector-predict=example_usage:main",
        ],
    },
    include_package_data=True,
    keywords="deepfake detection voice audio ai machine-learning cnn tensorflow",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/deepfake-voice-detector/issues",
        "Source": "https://github.com/yourusername/deepfake-voice-detector",
        "Documentation": "https://github.com/yourusername/deepfake-voice-detector/blob/main/README.md",
    },
)
