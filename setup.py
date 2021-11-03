import setuptools

setuptools.setup(
    name="sirc_image_utils",
    version='0.1',
    py_modules=['bulk_reproject'],
    install_requires=open('requirements.txt').readlines(),
    entry_points='''
        [console_scripts]
        bulk_reproject = scripts:bulk_reproject
    ''',
)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sirc_image_utils",
    version="0.0.1",
    author="Nicholas Wright",
    author_email="",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SeaIceResearchCenter/sirc_image_utils",
    project_urls={
        "Bug Tracker": "https://github.com/SeaIceResearchCenter/sirc_image_utils/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    
    entry_points='''
        [console_scripts]
        bulk_reproject = scripts:bulk_reproject
    ''',

    python_requires=">=3.6",
)