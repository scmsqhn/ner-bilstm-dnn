#!/bin/bash

#curname = $0
trorev=$1
#argcnt = $#

find ./ -type f -name "*.py" | xargs grep -s "$trorev" | tail -n 100
find ./ -type f -name "*.log" | xargs grep -s "$trorev" | tail -n 100
find ./ -type f -name "*.md" | xargs grep -s "$trorev" | tail -n 100
