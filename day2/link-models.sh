#!/bin/bash

mkdir -p ~/.keras/models/
ln -sf /wrk/mvsjober/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/

mkdir -p ~/.cache/torch/checkpoints/
ln -sf /wrk/mvsjober/vgg16-397923af.pth ~/.cache/torch/checkpoints/
