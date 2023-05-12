#!/bin/bash
python prod_single.py
torchrun prod.py --size 2
torchrun prod.py --size 4