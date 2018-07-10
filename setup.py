"""Setup script for pytorch_mask_rcnn."""
from setuptools import setup
from setuptools import find_packages


setup(
    name='pytorch_mask_rcnn',
    version='0.1',
    description='A PyTorch port of the Matterport Mask R-CNN',
    url='http://github.com/mjstevens777/pytorch-mask-rcnn',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        # Keep these in sync with requirements.txt
        'torch',
        'matplotlib',
        'scipy',
        'scikit-image',
        'h5py',
    ],
    package_data={
        'pytorch_mask_rcnn': [
            'nms/src/*.cpp',
            'nms/src/*.h',
            'nms/src/cuda/*.cu',
            'nms/src/cuda/*.h',
            'roialign/roi_align/src/*.cpp',
            'roialign/roi_align/src/*.h',
            'roialign/roi_align/src/cuda/*.cu',
            'roialign/roi_align/src/cuda/*.h',
        ]
    },
    zip_safe=False)
