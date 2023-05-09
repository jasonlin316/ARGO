#!/bin/bash
torchrun prod.py --size 1
torchrun prod.py --size 2
torchrun prod.py --size 4
torchrun cite2.py --size 1
torchrun cite2.py --size 2
torchrun cite2.py --size 4