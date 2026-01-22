# Wine ML Docker

This project is a machine learning application focused on predicting wine quality using a dataset of red wine characteristics. The application is containerized using Docker, allowing for easy deployment and execution in a consistent environment.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [CI/CD](#cicd)
- [License](#license)
- [Contributing](#contributing)

## Features
- Predicts wine quality based on various physicochemical tests.
- Containerized using Docker for consistent and reproducible environments.
- Automated CI/CD pipeline using GitHub Actions for continuous integration and deployment.

## Installation
To get started with the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Laxman52/wine-ml-docker.git
   cd wine-ml-docker
   ```

2. Build the Docker image:
   ```bash
   docker build -t wine-ml .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 5000:5000 wine-ml
   ```

## Usage
After running the Docker container, you can access the application by navigating to `http://localhost:5000` in your web browser. Follow the on-screen instructions to input data and receive predictions on wine quality.

## Dataset
The project uses the `winequality-red.csv` dataset, which contains various attributes related to red wine, including:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

## CI/CD
The project includes automated CI/CD pipelines set up using GitHub Actions. This ensures that any changes made to the codebase are automatically tested and deployed.

## License
This project does not have a specific license. Please check the repository for any updates.

## Contributing
Contributions are welcome! If you have suggestions for improvements or features, please feel free to create an issue or submit a pull request.

---
