#!/bin/bash

echo "🔄 Restarting memU services..."
docker-compose restart

echo ""
echo "✅ Services restarted!"
echo ""
echo "View logs with: docker-compose logs -f"
