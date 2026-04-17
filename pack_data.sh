#!/bin/bash                                                                                                                                                                                 
#PBS -l walltime=02:00:00                                                                                                                                                                   
#PBS -l filesystems=flare                                                                                                                                                                   
#PBS -q workq                                                                                                                                                                               
#PBS -A <your_project>                                                                                                                                                                      
module load frameworks                                                                                                                  
cd /flare/datascience/vsastry/projects/ptcho_vit/simple_pack_fm_ss/ptycho-vit 
                                                            
# 4 nodes × 12 ranks = 48 MPI ranks → 48 shards packed in parallel                                                                                                                          
# # Each rank writes ~33 shards (1563 shards / 48 ranks)                                                                                                                                      
mpiexec -n 96 --ppn 12 python pack_hdf5.py  

