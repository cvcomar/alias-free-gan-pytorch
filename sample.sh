#!/bin/bash

ckpt=$(ls ./checkpoint/ | sort -r | head -n 1)

if [ -z $ckpt ]; then
    echo "Error: no checkpoint specified"
fi

target=./generate_${1:-rotation}.py

echo target: $target

python $target ./checkpoint/$ckpt
