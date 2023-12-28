FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
ENV TZ Asia/Shanghai
ENV http_proxy http://router4.ustb-ai3d.cn:3128
ENV https_proxy http://router4.ustb-ai3d.cn:3128

RUN conda install -c torch-points3d torch-points-kernels