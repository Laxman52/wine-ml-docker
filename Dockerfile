# Use a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the Python script
COPY main.py .
COPY winequality-red.csv .


# Install the required dependencies
RUN pip install --no-cache-dir pandas scikit-learn matplotlib seaborn

# Set the entry point
ENTRYPOINT ["python", "main.py"]
