from setuptools import setup

setup(
    name='dmgr',
    version='0.0.2',
    packages=['dmgr'],
    url='',
    license='MIT',
    author='Filip Korzeniowski',
    author_email='filip.korzeniowski@jku.at',
    description='',
    requires=['numpy'],
    extras_require={
        'whitening': ['sklearn', 'scipy']
    }
)
