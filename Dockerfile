# Use the official Playwright image
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Run the server
CMD ["python", "main.py"]