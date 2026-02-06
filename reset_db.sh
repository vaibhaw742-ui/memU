#!/bin/bash
echo "🗑️  Resetting database..."
echo ""

# Stop services
echo "Stopping services..."
docker-compose down

# Remove postgres volume to completely reset the database
echo "Removing database volume..."
docker volume rm memu_postgres_data

# Start services
echo "Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for database to be ready..."
sleep 5

echo ""
echo "✅ Database reset complete!"
echo ""
echo "The database will be recreated with the current schema on next run."