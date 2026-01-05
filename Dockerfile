FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Dependances
COPY requirements.txt .
RUN pip install -r requirements.txt

# App source code
COPY api/ ./api
COPY src/ ./src
COPY models/ ./models
COPY data/ ./data

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]