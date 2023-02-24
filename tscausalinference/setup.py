from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Simplificación de la conection a el API REST de Adquality en Python.'

with open('README.md', encoding='utf-8') as f:
    long_description_english = f.read()

setup(
    name='tscausalinference',
    version=VERSION,
    description=DESCRIPTION,
    author='Carlos Trujillo',
    author_email='carlangastr@gmail.com',
    long_description = long_description_english,
    long_description_content_type = 'text/markdown',
    packages=find_packages(),
    keywords = ['python', 'causalimpact', 'causal inference', 'marketing', 'prophet', 'bootstrap'],
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'prophet',
        'scipy',
        'tabulate',
        'statsmodels'
        'seaborn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)