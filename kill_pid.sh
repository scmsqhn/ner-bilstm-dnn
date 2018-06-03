#!/bin/bash
kill -9 $(ps -ef | grep python | grep -v grep | awk '{print $2}')
#kill -9 $(ps -ef | grep vim | grep -v grep | awk '{print $2}')
kill -9 $(ps -ef | grep ps -af| grep -v grep | awk '{print $2}')
ps -af
free -h
