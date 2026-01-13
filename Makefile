.PHONY: help up down restart logs logs-f ps clean test test-cov lint format init-db kafka-topics

# Default target
help:
	@echo "IndustrialMind - Development Commands"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make up              - Start all Docker services"
	@echo "  make down            - Stop all Docker services"
	@echo "  make restart         - Restart all Docker services"
	@echo "  make logs            - Show all logs"
	@echo "  make logs-f          - Follow all logs"
	@echo "  make ps              - Show running containers"
	@echo "  make clean           - Remove all containers, volumes, and networks"
	@echo ""
	@echo "Database:"
	@echo "  make init-db         - Initialize databases with schemas"
	@echo "  make kafka-topics    - Create required Kafka topics"
	@echo ""
	@echo "Development:"
	@echo "  make test            - Run all tests"
	@echo "  make test-cov        - Run tests with coverage"
	@echo "  make lint            - Run code linting"
	@echo "  make format          - Format code with black/isort"
	@echo ""
	@echo "Services (when implemented):"
	@echo "  make simulator       - Start data simulator"
	@echo "  make ingestion       - Start data ingestion service"
	@echo "  make dashboard       - Start Streamlit dashboard"

# Infrastructure commands
up:
	@echo "Starting all Docker services..."
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@make ps

down:
	@echo "Stopping all Docker services..."
	docker-compose down

restart:
	@echo "Restarting all Docker services..."
	docker-compose restart

logs:
	docker-compose logs

logs-f:
	docker-compose logs -f

ps:
	docker-compose ps

clean:
	@echo "WARNING: This will remove all containers, volumes, and data!"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	docker-compose down -v --remove-orphans
	@echo "Cleanup complete"

# Database commands
init-db:
	@echo "Initializing databases..."
	@echo "Creating PostgreSQL schema..."
	docker-compose exec -T postgres psql -U admin -d industrialmind -f /docker-entrypoint-initdb.d/schema.sql || echo "Schema creation failed or already exists"
	@echo "Database initialization complete"

kafka-topics:
	@echo "Creating Kafka topics..."
	docker-compose exec kafka kafka-topics --create --if-not-exists \
		--bootstrap-server localhost:9092 \
		--topic sensor-readings \
		--partitions 3 \
		--replication-factor 1
	docker-compose exec kafka kafka-topics --create --if-not-exists \
		--bootstrap-server localhost:9092 \
		--topic anomaly-detected \
		--partitions 3 \
		--replication-factor 1
	docker-compose exec kafka kafka-topics --create --if-not-exists \
		--bootstrap-server localhost:9092 \
		--topic maintenance-predictions \
		--partitions 3 \
		--replication-factor 1
	docker-compose exec kafka kafka-topics --create --if-not-exists \
		--bootstrap-server localhost:9092 \
		--topic alerts \
		--partitions 3 \
		--replication-factor 1
	@echo "Kafka topics created"
	@echo "Listing topics:"
	docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Testing commands
test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=services --cov=shared --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Linting and formatting
lint:
	@echo "Running linters..."
	flake8 services/ shared/ tests/
	mypy services/ shared/ tests/
	@echo "Linting complete"

format:
	@echo "Formatting code..."
	black services/ shared/ tests/
	isort services/ shared/ tests/
	@echo "Formatting complete"

# Service commands (to be implemented)
simulator:
	@echo "Starting data simulator..."
	python services/data_platform/simulator/sensor_simulator.py

ingestion:
	@echo "Starting data ingestion service..."
	python services/data_platform/ingestion/kafka_consumer.py

dashboard:
	@echo "Starting Streamlit dashboard..."
	streamlit run services/dashboard/app.py

# Development setup
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "Dependencies installed"

setup: install up kafka-topics
	@echo "Development environment setup complete!"
	@echo ""
	@echo "Access services at:"
	@echo "  InfluxDB UI:    http://localhost:8086"
	@echo "  MLflow UI:      http://localhost:5000"
	@echo "  Grafana:        http://localhost:3000 (admin/admin)"
	@echo "  Prometheus:     http://localhost:9090"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Implement data simulator"
	@echo "  2. Implement data ingestion"
	@echo "  3. Implement dashboard"
