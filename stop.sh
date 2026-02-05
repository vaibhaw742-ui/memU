#!/bin/bash

echo "🛑 Stopping memU services..."
docker-compose stop

echo ""
echo "✅ Services stopped!"
echo ""
echo "Note: Data is preserved. Use ./start.sh to restart."
