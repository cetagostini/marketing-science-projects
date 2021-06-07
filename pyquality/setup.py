# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 03:34:26 2021

@author: Carlos Trujillo
"""

from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.4'
DESCRIPTION = 'Simplificación de la conection a el API REST de Adquality en Python.'

setup(
      name = 'py_quality',
      version = VERSION,
      author = 'Carlangastr',
      auhor_email = '',
      description = DESCRIPTION,
      long_description_content_type = 'text/markdown',
      packages = find_packages(),
      install_requires = ['pandas'],
      keywords = ['python', 'adquality', 'sql', 'marketing', 'audience'],
      classifiers = [
          'Development Status :: 1 - Planning',
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3',
          'Operating System :: Unix',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          ]

      )