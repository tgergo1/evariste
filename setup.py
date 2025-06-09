from setuptools import setup, find_packages

setup(
    name="evariste",
    version="0.1.0",
    description="A bold approach to general artificial intelligence",
    author="Evariste Team",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyqtgraph",
        "PyQt5",
    ],
    entry_points={
        'console_scripts': [
            'evariste=evariste.main:main',
        ],
    },
)
