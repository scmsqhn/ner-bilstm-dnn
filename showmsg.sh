#!/bin/bash
find ./ -name bilstm_logger.log | xargs grep -s "mean" | tail -n 10
find ./ -name bilstm_logger.log | xargs grep -s "perenal" | tail -n 10
find ./ -name bilstm_logger.log | xargs grep -s "_acc" | tail -n 10
