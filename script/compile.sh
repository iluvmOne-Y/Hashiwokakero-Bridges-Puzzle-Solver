if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    python3 -O source/main.py
elif [[ "$OSTYPE" == "win32"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    python -O source/main.py
fi

# Recursively remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +