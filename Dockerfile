# Use slim Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY main.py .
COPY winequality-red.csv .

# Install dependencies
RUN pip install --no-cache-dir pandas scikit-learn matplotlib seaborn joblib

# Run the script
CMD ["python", "main.py"]
