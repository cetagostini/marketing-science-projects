# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 03:34:26 2021

@author: Carlos Trujillo
"""

from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.2'
DESCRIPTION = 'Simplification of the connection to the Admetricks REST API in Python.'

with open('english_readme.md', encoding='utf-8') as f:
    long_description_english = f.read()


setup(
      name = 'py_metricks',
      version = VERSION,
      author = 'Carlangastr',
      auhor_email = '',
      description = DESCRIPTION,
      long_description=long_description_english,
      long_description_content_type = 'text/markdown',
      packages = find_packages(),
      install_requires = ['pandas', 'ftfy', 'requests', 'unidecode'],
      keywords = ['python', 'admetricks', 'sql', 'marketing', 'audience'],
      classifiers = [
          'Development Status :: 1 - Planning',
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3',
          'Operating System :: Unix',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          ]

      )