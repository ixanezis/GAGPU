#!/bin/bash -eu
nvcc -ccbin g++-4.6 -arch=sm_20 --use_fast_math GA-shared.cu -o GA-shared
