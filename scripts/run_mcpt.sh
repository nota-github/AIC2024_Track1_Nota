#!/bin/bash

# run mcpt on test set
python3 eval_mcpt.py

# merge results for submission
python3 tools/merge_results.py