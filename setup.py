from setuptools import setup

setup(
    name='Vertex AI 4 Dummies',
    version='0.1.0',    
    description='A package to make deploying on Vertex AI easier',
    url='https://github.com/jamesonhohbein/vertex_deployment_4dummies',
    author='Jameson Hohbein',
    author_email='jhohbein@gmail.com',
    license='BSD 2-clause',
    packages=['vertex4dummies',],
    install_requires=['google-cloud-aiplatform>=1.15.1',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.7',
    ],
)