#!/bin/bash
. ../utils.sh

sudo docker build -t=h4d_base -f h4d_base_docker .

rm -rf h4d
rm -rf experiments