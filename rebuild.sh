#!/bin/bash
echo "🔨 Rebuilding memU application..."
echo ""

# Stop only the memu service
echo "Stopping memU app..."
docker-compose stop memu

# Remove the memu container
echo "Removing old memU container..."
docker-compose rm -f memu

# Remove the old image to force a complete rebuild
echo "Removing old memU image..."
docker-compose images -q memu | xargs -r docker rmi -f

# Rebuild only the memu service with no cache
echo "Building new memU image..."
docker-compose build --no-cache memu

# Start services (postgres should already be running)
echo "Starting memU app..."
docker-compose up -d

echo ""
echo "✅ Rebuild complete!"
echo ""
echo "View logs with: docker-compose logs -f memu"