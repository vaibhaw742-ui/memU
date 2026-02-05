#!/bin/bash

echo "🚀 Starting memU services..."
docker-compose up -d

echo ""
echo "✅ Services started!"
echo ""
echo "View logs with: docker-compose logs -f"
echo "Check status with: docker-compose ps"
