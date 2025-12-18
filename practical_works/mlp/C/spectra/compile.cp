#!/bin/sh

gcc -O3 -Wno-unused-result ../ext_module.c spectra_online.c -o spectra_online -lm
gcc -O3 -Wno-unused-result ../ext_module.c spectra_sgd.c -o spectra_sgd -lm

#uncoment the following after you installed openblas
gcc -O3 -Wno-unused-result ../ext_module_blas.c spectra_batch.c -o spectra_batch -lm -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas 
gcc -O3 -Wno-unused-result ../ext_module_blas.c spectra_rebalance.c -o spectra_rebalance -lm -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas 


