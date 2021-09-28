#!/bin/bash

for ckpt in $(find ./checkpoint/ | grep pt | sort -r | head -n 17 | sort | head -n 14); do
    echo $ckpt
    echo $ckpt >> ./fid-log
    rm -rf ./img4fid/
    python ./generate4fid.py $ckpt
    python -m pytorch_fid ./img4fid/ ~/dataset-local/ffhq/fid256/ >> ./fid-log
done
