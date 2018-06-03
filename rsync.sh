#!/bin/bash

# rsync [OPTION]... SRC DEST
# rsync [OPTION]... SRC [USER@]host:DEST
# rsync [OPTION]... [USER@]HOST:SRC DEST
# rsync [OPTION]... [USER@]HOST::SRC DEST
# rsync [OPTION]... SRC [USER@]HOST::DEST
# rsync [OPTION]... rsync://[USER@]HOST[:PORT]/SRC [DEST]

rsync -vzrtopg -e 'ssh -p 15002' --progress  ~/bilstm/  distdev@113.204.229.74:~/bilstm.bak/
