#!/bin/bash

echo "🔨 Rebuilding memU application..."
echo ""

# Stop only the memu service
echo "Stopping memU app..."
docker-compose stop memu

# Remove the memu container
echo "Removing old memU container..."
docker-compose rm -f memu

# Rebuild only the memu service
echo "Building new memU image..."
docker-compose build --no-cache memu

# Start services (postgres should already be running)
echo "Starting memU app..."
docker-compose up -d

echo ""
echo "✅ Rebuild complete!"
echo ""
echo "View logs with: docker-compose logs -f memu"
