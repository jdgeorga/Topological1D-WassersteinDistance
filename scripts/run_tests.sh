#!/bin/bash
set -e

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install test dependencies
pip install -e ".[test]"

# Run tests with coverage
pytest

# Print coverage report
coverage report

# Generate HTML coverage report
coverage html

echo "Tests completed. See htmlcov/index.html for coverage report."
