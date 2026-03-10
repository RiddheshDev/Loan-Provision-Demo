FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only required files and folders
COPY app.py .
COPY prod_requirements.txt .
COPY src/ ./src/
# Install dependencies
RUN pip install --no-cache-dir -r prod_requirements.txt

# Expose port (optional, if your app runs a server)
EXPOSE 8000

# Command to run your application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["python","app.py"]

