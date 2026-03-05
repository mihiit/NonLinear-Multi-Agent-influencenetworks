from setuptools import setup, find_packages

setup(
    name="nonlinear_influence_networks",
    version="1.0.0",
    author="Mihit Nanda, Hannah Nagpall",
    author_email="mihit.nanda.cs27@iilm.edu",
    description=(
        "Decoupling Transient Instability and Steady-State Persistence "
        "in Nonlinear Multi-Agent Influence Networks"
    ),
    url="https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "networkx>=3.0",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "jupyter>=1.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
