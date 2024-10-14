from setuptools import setup, find_packages

# Try to read the long description from README.md
try:
    with open('README.md', 'r', encoding='utf-8') as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = 'A library for data engineering tasks. Created for the HS2024 Data Science Project Traffic Status.'

setup(
    name='DataEngineeringLibrary',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas', 'numpy'
    ],
    author='Samuel Paul',
    author_email='samuel.paul6314@gmail.com',
    description='A library for data engineering tasks. Created for the HS2024 Data Science Project Traffic Status.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/data_engineering_library',  # Replace with your GitHub repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)