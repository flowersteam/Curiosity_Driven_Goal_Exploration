from setuptools import setup

setup(name='latentgoalexplo',
      version='0.0.1',
      description='Goal exploration using latent representation',
      url='https://github.com/flowersteam/Curiosity_Driven_Goal_Exploration',
      author='Adrien Laversanne-Finot',
      author_email='adrien.laversanne-finot@inria.fr',
      license='MIT',
      packages=['latentgoalexplo'],
      install_requires=[
          'numpy',
          'torch',
          'explauto',
          'gizeh',
          'matplotlib',
          'visdom',
      ],
      zip_safe=False)