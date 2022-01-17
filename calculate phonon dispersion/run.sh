#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`

startTime_s=`date +%s`

echo "   "
echo "******************Start running GPUMD tasks********************"
echo "   "

export CUDA_VISIBLE_DEVICES=0

/home/liangting/liangting/GPUMD-2.9/src/gpumd < run_phonon_dispersion.txt

nvidia-smi

endTime=`date +%Y%m%d-%H:%M`

endTime_s=`date +%s`

sumTime=$[$endTime_s - $startTime_s]

Hour=$[sumTime/3600]

echo "   "
echo "******************End running GPUMD tasks********************"
echo "   "

echo "$startTime ---> $endTime" "Total:$Hour  hours"
