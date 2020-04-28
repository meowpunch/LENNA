#!/bin/bash
python3 cifar_arch_search_recasting.py --student_path 'wo_warmup_combined/' --warmup_epochs 0 --gpu '1' --num_blocks '1, 1, 1' --num_layers 4 --n_epochs 200 --clear_recasting_student True --mixedge_ver 2 --increase_term 5 | tee wo_warmup_combined/full.log
