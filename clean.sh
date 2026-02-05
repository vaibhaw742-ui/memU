#!/bin/bash

echo "⚠️  WARNING: This will remove all containers, volumes, and data!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
    echo "🧹 Cleaning up memU..."
    docker-compose down -v
    echo ""
    echo "✅ Cleanup complete! All data has been removed."
else
    echo "❌ Cleanup cancelled."
fi
