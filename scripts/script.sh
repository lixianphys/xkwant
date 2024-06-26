#!/bin/bash

# Run the first Python script
echo "Running dirachbar.py..."
python3.8 dirachbar.py
if [ $? -ne 0 ]; then
    echo "dirachbar.py failed. Continuing."
fi

# Run the second Python script
echo "Running rashbahbar.py..."
python3.8 rashbahbar.py
if [ $? -ne 0 ]; then
    echo "rashbahbar.py failed. Continuing."
fi

# Run the third Python script
echo "Running quadhbar.py..."
python3.8 quadhbar.py
if [ $? -ne 0 ]; then
    echo "quadhbar.py failed"
fi

echo "Exiting."
