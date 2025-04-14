FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app .

# Expose port
EXPOSE 8080

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
