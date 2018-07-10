#include <torch/torch.h>
#include <THC/THC.h>
#include "cuda/crop_and_resize_kernel.h"


void crop_and_resize_gpu_forward(
    at::Tensor image,
    at::Tensor boxes,           // [y1, x1, y2, x2]
    at::Tensor box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    at::Tensor crops
) {
    const int batch_size = image.size(0);
    const int depth = image.size(1);
    const int image_height = image.size(2);
    const int image_width = image.size(3);

    const int num_boxes = boxes.size(0);

    // init output space
    crops.resize_({num_boxes, depth, crop_height, crop_width});
    crops.zero_();

    CropAndResizeLaucher(
        image.data<float>(),
        boxes.data<float>(),
        box_index.data<int>(),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth, extrapolation_value,
        crops.data<float>()
    );
}


void crop_and_resize_gpu_backward(
    at::Tensor grads,
    at::Tensor boxes,      // [y1, x1, y2, x2]
    at::Tensor box_index,    // range in [0, batch_size)
    at::Tensor grads_image // resize to [bsize, c, hc, wc]
) {
    // shape
    const int batch_size = grads_image.size(0);
    const int depth = grads_image.size(1);
    const int image_height = grads_image.size(2);
    const int image_width = grads_image.size(3);

    const int num_boxes = grads.size(0);
    const int crop_height = grads.size(2);
    const int crop_width = grads.size(3);

    // init output space
    grads_image.zero_();

    CropAndResizeBackpropImageLaucher(
        grads.data<float>(),
        boxes.data<float>(),
        box_index.data<int>(),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth,
        grads_image.data<float>()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "crop_and_resize_gpu_forward", &crop_and_resize_gpu_forward, R"docstring(
      Crop and resize forward
    )docstring",
    py::arg("image"), py::arg("boxes"), py::arg("box_index"),
    py::arg("extrapolation_value"), py::arg("crop_height"),
    py::arg("crop_width"), py::arg("crops"));
  m.def(
    "crop_and_resize_gpu_backward", &crop_and_resize_gpu_backward, R"docstring(
      Crop and resize backward
    )docstring",
    py::arg("grads"), py::arg("boxes"), py::arg("box_index"),
    py::arg("grads_image"));
}
