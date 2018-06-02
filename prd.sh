#!/bin/bash

echo "> TRAIN or EVAL"
#curname = $0
trorev=$1
#argcnt = $#

cd /home/distdev/bilstm
. activate bilstm
./kill_pid.sh

pyflakes bilstm_train.py
pyflakes datahelper.py

echo $trorev

if [ $trorev -eq 5 ]
  then 
    #echo $trorev
    nohup python bilstm_train.py > tmp 2>&1 &
    echo "nohup python bilstm_train.py > tmp 2>&1 &"
elif [ $trorev -eq 2 ]
  then
    #echo $trorev
    nohup python eval_bilstm.py > tmp 2>&1 &
    echo "nohup python eval_bilstm.py > tmp 2>&1 &"
elif [ $trorev -eq 3 ]
  then
    #echo $trorev
    python bilstm_train.py
    echo "python train_bilstm.py"
elif [ $trorev -eq 4 ]
  then
    #echo $trorev
    python eval_bilstm.py
    echo "python eval_bilstm.py"
elif [ $trorev -eq 0 ]
  then
    #echo $trorev
    echo "> 0 to train /  1 to eval"
else
  echo "para wrong not tr or ev"
fi

echo "> FINISH"

