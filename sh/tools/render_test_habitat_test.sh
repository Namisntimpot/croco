#!/bin/bash
#
tlaunch --gpu=1 --cpu=4 --memory=32768 --positive-tag=4090 -- python -m datasets.habitat_sim.generate_from_metadata --metadata_filename ../data/habitat-sim-data/tmp_metadata/habitat-test-scenes/van-gogh-room/metadata.json --output_dir ../data/habitat-sim-data/pairs_dataset/habitat-test-scenes/van-gogh-room
