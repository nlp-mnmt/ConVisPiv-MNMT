#!/bin/bash


python3 scripts/average_checkpoints.py \
			--inputs results/pre_mixup/mmtimg \
			--num-epoch-checkpoints 10 \
			--output results/pre_mixup/mmtimg/model.pt \



			