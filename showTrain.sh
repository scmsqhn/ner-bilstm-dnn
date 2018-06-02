#!/bin/bash
#find ./ -name bilstm_logger.log | xargs grep -s "mean_acc" | tail -n 30
find ./ -name bilstm_logger.log | xargs grep -s "res" | tail -n 50 
