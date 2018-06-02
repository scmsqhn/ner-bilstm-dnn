#!/bin/bash
find ~/** -name "*.py" -type f | xargs grep -s "def sparse_softmax_cross_entropy_with_logit"


