#!/bin/bash

torchrun PyG/DDP-obgn-product-sage.py --process 1 --l_core 2
torchrun PyG/DDP-obgn-product-sage.py --process 1 --l_core 4
torchrun PyG/DDP-obgn-product-sage.py --process 2 --l_core 2
torchrun PyG/DDP-obgn-product-sage.py --process 2 --l_core 4
torchrun PyG/DDP-obgn-product-sage.py --process 4 --l_core 2
torchrun PyG/DDP-obgn-product-sage.py --process 4 --l_core 4
