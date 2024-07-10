from setuptools import setup, find_packages

setup(
    name="NextGenJAX",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A JAX-based neural network library surpassing Google DeepMind's Haiku and Optax",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VishwamAI/NextGenJAX",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "jax==0.4.27",
        "jaxlib==0.4.27",
        "flax==0.8.5",
        "optax==0.2.3",
        "numpy==1.26.4",
        "scipy==1.10.1",
    ],
    extras_require={
        "dev": [
            "pytest==6.2.4",
            "flake8==3.9.2",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
