// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
#include "nms_cuda.h"
#include <torch/torch.h>
#include <THC/THC.h>
#include <TH/TH.h>
#include <math.h>
#include <stdio.h>

#include "cuda/nms_kernel.h"


int gpu_nms(at::Tensor keep, at::Tensor num_out, at::Tensor boxes, float nms_overlap_thresh) {
  // boxes has to be sorted

  // Number of ROIs
  int boxes_num = boxes.size(0);
  int boxes_dim = boxes.size(1);

  float* boxes_flat = boxes.contiguous().data<float>();

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  at::Tensor mask = at::empty(at::CUDA(at::kLong), {boxes_num, col_blocks});
  long* mask_flat = mask.data<long>();

  _nms(boxes_num, boxes_flat, mask_flat, nms_overlap_thresh);

  at::Tensor mask_cpu = mask.toBackend(at::Backend::CPU);

  long * mask_cpu_flat = mask_cpu.data<long>();

  at::Tensor remv_cpu = at::zeros(at::CPU(at::kLong), {col_blocks});
  long* remv_cpu_flat = remv_cpu.data<long>();

  long * keep_flat = keep.data<long>();
  long num_to_keep = 0;

  int i, j;
  for (i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv_cpu_flat[nblock] & (1ULL << inblock))) {
      keep_flat[num_to_keep++] = i;
      long *p = &mask_cpu_flat[0] + i * col_blocks;
      for (j = nblock; j < col_blocks; j++) {
        remv_cpu_flat[j] |= p[j];
      }
    }
  }

  long * num_out_flat = num_out.data<long>();
  * num_out_flat = num_to_keep;

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "gpu_nms", &gpu_nms, R"docstring(
      NMS
    )docstring",
    py::arg("keep"), py::arg("num_out"), py::arg("boxes"),
    py::arg("nms_overlap_thresh"));
}
