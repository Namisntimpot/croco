#!/bin/bash

tlaunch --gpu=1 --cpu=4 --memory=32768 --positive-tag=4090 -- python -m datasets.habitat_sim.generate_from_metadata --metadata_filename ../data/habitat-sim-data/tmp_metadata_replica_cad/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_02/metadata.json --output_dir ../data/habitat-sim-data/pairs_dataset/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_02
