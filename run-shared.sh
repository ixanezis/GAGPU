#!/bin/bash -eu
nvcc -ccbin g++-4.6 -arch=sm_20 rosenbrick-shared.cu -o rosenbrick-shared
