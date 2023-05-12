#!/bin/bash
python cite2_single.py
torchrun cite2.py --size 2
torchrun cite2.py --size 4