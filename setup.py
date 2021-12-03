import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cs107_salad",
    version="0.0.7",
    author="Neil Sehgal, Andrew Zhang, Diwei Zhang, Lotus Xia",
    author_email="neil_sehgal@g.harvard.edu, andrew_zhang@college.harvard.edu, diwei_zhang@hsph.harvard.edu, lxia@g.harvard.edu",
    description="Auto-Differentiation Program",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cs107-DaLand/cs107-FinalProject",
    project_urls={
        "Bug Tracker": "https://github.com/cs107-DaLand/cs107-FinalProject/issues",
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'numpy'],
)
