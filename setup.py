from setuptools import setup, find_packages

setup(name='latentgoalexplo',
      version='0.0.1',
      description='Goal exploration using latent representation',
      url='https://github.com/flowersteam/Curiosity_Driven_Goal_Exploration',
      author='Adrien Laversanne-Finot',
      author_email='adrien.laversanne-finot@inria.fr',
      license='MIT',
      packages=[package for package in find_packages()
                if package.startswith('latentgoalexplo')],
      install_requires=[
          'numpy',
          'torch',
          'explauto',
          'gizeh',
          'matplotlib',
          'visdom',
      ],
      include_package_data=True,
      zip_safe=False)
