#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <python_script>"
    echo ""
    echo "Examples:"
    echo "  ./run.sh tests/test_inmemory.py"
    echo "  ./run.sh tests/test_postgres.py"
    echo "  ./run.sh examples/example_1_conversation_memory.py"
    exit 1
fi

echo "🐍 Running: $1"
echo ""
docker-compose exec memu .venv/bin/python "$1"

