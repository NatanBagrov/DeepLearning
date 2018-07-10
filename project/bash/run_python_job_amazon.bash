#!/bin/bash

source ~/tensorflow/bin/activate
echo "Adding `pwd` to python environment"
export PYTHONPATH="`pwd`:${PYTHONPATH}"
timestamp=`date`
echo "Time stamp is ${timestamp}"
python3 $@ > "${timestamp}.out" 2>"${timestamp}.err" &
pid=$!
echo "Pid is ${pid}"
wait
