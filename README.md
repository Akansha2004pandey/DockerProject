# Sentiment Analysis with Docker

This repository contains a **Sentiment Analysis** project where we built a machine learning model to analyze the sentiment of text data (positive, negative, or neutral). The project is containerized using Docker, and the model is exposed via an API using **FastAPI**. We have set up a **CI/CD pipeline** to automatically build and push the Docker image to **Docker Hub** upon code changes.

## Project Structure


## Technologies Used

- **Machine Learning**: Scikit-learn, joblib (for loading the model)
- **API Framework**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Registry**: Docker Hub

## Steps to Run Locally

### Prerequisites

1. **Docker**: Install Docker to build and run containers locally.
2. **Python**: If you want to run FastAPI locally before Dockerizing, ensure you have Python 3.9+ and install dependencies from `requirements.txt`.

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/DockerProject.git
cd DockerProject
