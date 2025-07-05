#!/bin/bash

# AMEVulDetector runner script
# This script sets up the environment and runs the AMEVulDetector with various options

# Activate virtual environment
source venv/bin/activate

echo "Running AMEVulDetector with default settings (EncoderWeight)..."
python3 AMEVulDetector.py

# echo -e "\n" + "="*50 + "\n"

# echo "Running AMEVulDetector with EncoderWeight model and custom hyperparameters..."
# python3 AMEVulDetector.py --model EncoderWeight --lr 0.002 --dropout 0.2 --epochs 10 --batch_size 32

# echo -e "\n" + "="*50 + "\n"

# echo "Running AMEVulDetector with EncoderAttention model..."
# python3 AMEVulDetector.py --model EncoderAttention --lr 0.001 --epochs 10 --batch_size 16

# echo -e "\n" + "="*50 + "\n"

# echo "Running AMEVulDetector with FNNModel..."
# python3 AMEVulDetector.py --model FNNModel --lr 0.003 --epochs 10 --batch_size 64

echo "All experiments completed!"