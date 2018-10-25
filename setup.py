from setuptools import setup

requirements = [
    # TODO: put your package requirements here
]

test_requirements = [
    'pytest',
    'pytest-cov',
    'pytest-faulthandler',
    'pytest-mock',
    'pytest-qt',
    'pytest-xvfb',
]

setup(
    name='foct-viewer',
    version='0.0.1',
    description="a simple viewer for Optovue foct files",
    author="Alexander Tomlinson",
    author_email='mexander@gmail.com',
    url='https://github.com/awctomlinson/foct-viewer',
    packages=['foct_viewer', 'foct_viewer.images',
              'foct_viewer.tests'],
    package_data={'foct_viewer.images': ['*.png']},
    entry_points={
        'console_scripts': [
            'FoctViewer=foct_viewer.foctviewer:main'
        ]
    },
    install_requires=requirements,
    zip_safe=False,
    keywords='foct-viewer',
    classifiers=[
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
