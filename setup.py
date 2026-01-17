"""
Setup script for huaju4k video enhancement tool.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text(encoding="utf-8").strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="huaju4k",
    version="0.1.0",
    description="Theater Video Enhancement Tool - Transform theater drama videos to 4K with specialized audio optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="huaju4k Development Team",
    author_email="dev@huaju4k.com",
    url="https://github.com/huaju4k/huaju4k",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'gpu': ['torch', 'torchvision'],
        'audio': ['librosa', 'soundfile'],
        'dev': ['pytest', 'pytest-cov', 'black', 'flake8', 'mypy'],
        'test': ['hypothesis', 'pytest-mock']
    },
    entry_points={
        'console_scripts': [
            'huaju4k=huaju4k.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Video :: Conversion",
    ],
    python_requires=">=3.8",
    keywords="video enhancement, 4K upscaling, theater, audio processing, AI upscaling",
    project_urls={
        "Bug Reports": "https://github.com/huaju4k/huaju4k/issues",
        "Source": "https://github.com/huaju4k/huaju4k",
        "Documentation": "https://huaju4k.readthedocs.io/",
    },
)