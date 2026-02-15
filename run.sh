#!/bin/bash
set -e

echo "=== Checking Ollama availability... ==="
until curl -s http://host.docker.internal:11434 > /dev/null; do
  echo "Waiting for Ollama..."
  sleep 2
done
echo "=== Ollama is available ==="

echo "=== Launching Flask ==="
# Run Flask in the foreground and log errors
python3 -u -m app.server || { 
    echo "Flask exited unexpectedly with code $?"; 
    sleep 10; 
}
