# Use official Python image as base
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Flask API on port 8080
EXPOSE 8080

# Command to run the Flask app
CMD ["python", "app.py"]
