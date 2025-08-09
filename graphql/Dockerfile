# Dockerfile for Strawberry GraphQL API
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY graphql_app.py ./

# Expose port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "graphql_app:app", "--host", "0.0.0.0", "--port", "8000"]
