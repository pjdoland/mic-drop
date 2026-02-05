"""mic-drop — local voice-cloning TTS pipeline."""

from setuptools import find_packages, setup

setup(
    name="mic-drop",
    version="0.1.0",
    description="Local voice-cloning TTS: Tortoise TTS + RVC in one pipeline",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="mic-drop",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        # Heavy ML backends.
        # NOTE: the RVC group (rvc-python, fairseq …) requires pip==24.0
        #       at install time.  Prefer  ./setup.sh  or the manual steps
        #       in requirements-rvc.txt over  pip install -e ".[ml]".
        "ml": [
            "tortoise-tts>=0.1.0",
            "rvc-python>=0.1.0",
            "fairseq>=0.12.0",
            "librosa>=0.10.0",
            "resampy>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mic-drop=tts_pipeline.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Audio",
    ],
)
