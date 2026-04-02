#!/bin/bash
# Start script for Cab Demand Prediction Backend
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/homebrew/opt/libomp/lib
source venv/bin/activate
python3 Backend/API.py
