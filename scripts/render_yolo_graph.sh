#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate python environment if needed (assuming user's environment)
# source activate ... 

echo "Checking for graphviz..."
if ! python -c "import graphviz" &> /dev/null; then
    echo "graphviz python package not found. Installing..."
    pip install graphviz
fi

if ! command -v dot &> /dev/null; then
    echo "Error: 'dot' command not found. Please install graphviz system package."
    echo "  Ubuntu: sudo apt-get install graphviz"
    echo "  Conda: conda install graphviz"
    exit 1
fi

echo "Running visualization script..."
python "${SCRIPT_DIR}/render_network_graph.py"

echo "Done."

