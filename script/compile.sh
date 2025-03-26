if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    python3 -O source/main.py
elif [[ "$OSTYPE" == "win32"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    python -O source/main.py
fi