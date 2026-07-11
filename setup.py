from setuptools import setup, find_packages

setup(
    name="geomtok",
    version="1.1.0",
    description="GeomTok — geometry-native tokenization for vector graphics",
    author="Byun",
    author_email="igotthepower0128@gmail.com",
    packages=find_packages(include=["geomtok", "geomtok.*"]),
    # 동봉 vocab 매니페스트(L2 머지 포함) — 결정성·오프라인 백킹
    package_data={"geomtok": ["data/*.json"]},
    include_package_data=True,
    python_requires=">=3.9",
    # OSS 코어는 numpy 만으로 완전 동작 (파싱·정규화·L1/L2·코덱·FSA 디코드)
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        # 토큰 경제 측정용 텍스트 BPE 베이스라인
        "eval": [
            "tiktoken>=0.5",
            "cairosvg>=2.7",
            "pillow>=9.0",
            "scikit-image>=0.20",
        ],
        # 매니지드 API 서버
        "server": [
            "fastapi>=0.100",
            "uvicorn[standard]>=0.23",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "httpx",
        ],
        # Phase 2 생성 모델 (토이 스케일, 비목표) — torch 의존
        "ml": [
            "torch>=2.0",
            "transformers>=4.30",
        ],
    },
    entry_points={
        "console_scripts": [
            "geomtok-serve=geomtok.server.app:_cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
)
