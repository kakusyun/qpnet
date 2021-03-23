from setuptools import setup, find_packages

setup(name="qpnet-pytorch",
      packages=find_packages(exclude=['.txt', '.jpg', '.jpeg', '.bmp',
                                      '.pkl', '.xml', '.xlsx', '.csv', '.png',
                                      'docs', '.h5', '.hdf5', 'tmp']))
