from setuptools import setup, find_packages


setup(name='moketools',
      version='0.0.1',
      description=u"Tools associated with MOKE experiment",
      classifiers=[],
      keywords='',
      author=u"Julian Irwin",
      author_email='julian.irwin@gmail.com',
      url='',
      license='MIT',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[],
      extras_require={
          'test': ['nose'],
      }
      # entry_points="""
      # [console_scripts]
      # pyskel=pyskel.scripts.cli:cli
      # """
      )
