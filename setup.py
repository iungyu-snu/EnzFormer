from setuptools import setup, find_packages

setup(
    name="EnzFormer",  
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],  
    entry_points={
        'console_scripts': [
            'train = train:main',
            'evaluate = evaluate:main',
            'embedding = embedding:main',
            'extract_dm = extract_dm:main',
        ],
    },
)

