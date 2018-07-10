import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.cpp_extension as cpp_extension
import os

_cpu_backend = cpp_extension.load(
    "cpu_crop_and_resize", [
        os.path.join(os.path.dirname(__file__), "src/crop_and_resize.cpp")
    ])

if torch.cuda.is_available():
    arch = os.getenv('CUDA_ARCH')
    if arch:
        extra_cuda_cflags = ['-arch', arch]
    else:
        extra_cuda_cflags = []
    _gpu_backend = cpp_extension.load(
        "gpu_crop_and_resize", [
            os.path.join(os.path.dirname(__file__), "src/crop_and_resize_gpu.cpp"),
            os.path.join(os.path.dirname(__file__), "src/cuda/crop_and_resize_kernel.cu"),
        ], extra_cuda_cflags=extra_cuda_cflags)


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros_like(image)

        if image.is_cuda:
            _gpu_backend.crop_and_resize_gpu_forward(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width, crops)
        else:
            _cpu_backend.crop_and_resize_forward(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width, crops)

        # save for backward
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)

        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)

        if grad_outputs.is_cuda:
            _gpu_backend.crop_and_resize_gpu_backward(
                grad_outputs, boxes, box_ind, grad_image
            )
        else:
            _cpu_backend.crop_and_resize_backward(
                grad_outputs, boxes, box_ind, grad_image
            )

        return grad_image, None, None


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)
