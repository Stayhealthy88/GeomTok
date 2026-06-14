from setuptools import setup, find_packages

setup(
    name="geomtok",
    version="0.7.0",
    description="GeomTok — geometry-native tokenization for vector graphics",
    author="Byun",
    author_email="igotthepower0128@gmail.com",
    packages=find_packages(include=["geomtok", "geomtok.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "ml": [
            "torch>=2.0",
            "transformers>=4.30",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
