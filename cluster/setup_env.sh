#!/bin/bash
module load cuda/11.5.0
module load python/3.11.7--gcc--10.2.0
export CPATH=/g100/home/userexternal/rberni00/spack-0.21.0-05/install/linux-centos8-skylake_avx512/gcc-8.4.1/brotli-1.0.9-iixj3lmxircel7apx463cyow5zj7iq5y/include:$CPATH
export LIBRARY_PATH=/g100/home/userexternal/rberni00/spack-0.21.0-05/install/linux-centos8-skylake_avx512/gcc-8.4.1/brotli-1.0.9-iixj3lmxircel7apx463cyow5zj7iq5y/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/g100/home/userexternal/rberni00/spack-0.21.0-05/install/linux-centos8-skylake_avx512/gcc-8.4.1/brotli-1.0.9-iixj3lmxircel7apx463cyow5zj7iq5y/lib64:$LD_LIBRARY_PATH
