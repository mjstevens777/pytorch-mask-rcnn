import torch
import numpy as np
import torch.utils.cpp_extension
import os

_cpu_nms = torch.utils.cpp_extension.load(
    'cpu_nms', [
        os.path.join(os.path.dirname(__file__), 'src/nms.cpp')
    ])

if torch.cuda.is_available():
    arch = os.getenv('CUDA_ARCH')
    if arch:
        extra_cuda_cflags = ['-arch', arch]
    else:
        extra_cuda_cflags = []
    _gpu_nms = torch.utils.cpp_extension.load(
        'gpu_nms', [
            os.path.join(os.path.dirname(__file__), 'src/nms_cuda.cpp'),
            os.path.join(os.path.dirname(__file__), 'src/cuda/nms_kernel.cu')
        ], extra_cuda_cflags=extra_cuda_cflags)

def pth_nms(dets, thresh):
  """
  dets has to be a tensor
  """
  if not dets.is_cuda:
    x1 = dets[:, 1]
    y1 = dets[:, 0]
    x2 = dets[:, 3]
    y2 = dets[:, 2]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.numpy().argsort()[::-1])).long()

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    _cpu_nms.cpu_nms(keep, num_out, dets, order, areas, thresh)

    return keep[:num_out[0]]
  else:
    x1 = dets[:, 1]
    y1 = dets[:, 0]
    x2 = dets[:, 3]
    y2 = dets[:, 2]
    scores = dets[:, 4]

    dets_temp = torch.FloatTensor(dets.size()).cuda()
    dets_temp[:, 0] = dets[:, 1]
    dets_temp[:, 1] = dets[:, 0]
    dets_temp[:, 2] = dets[:, 3]
    dets_temp[:, 3] = dets[:, 2]
    dets_temp[:, 4] = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.cpu().numpy().argsort()[::-1])).long().cuda()

    dets = dets[order].contiguous()

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    # keep = torch.cuda.LongTensor(dets.size(0))
    # num_out = torch.cuda.LongTensor(1)
    _gpu_nms.gpu_nms(keep, num_out, dets_temp, thresh)

    return order[keep[:num_out[0]].cuda()].contiguous()
    # return order[keep[:num_out[0]]].contiguous()

