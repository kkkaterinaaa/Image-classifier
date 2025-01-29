# Image Classification with FastAPI and Streamlit

This project implements an image classification system using a ResNet model and CIFAR-10 dataset. The backend is powered by **FastAPI** and the frontend by **Streamlit**. Users can upload images via a Streamlit web interface, which sends the images to a FastAPI backend that returns the predicted class.

## Features

- **FastAPI API** to serve image classification predictions using a ResNet model.
- **Streamlit Web App** for users to upload images and view predictions.
- **Dockerized Deployment** using Docker Compose for easy setup and management.

## Project Structure

```bash
├── code
│   ├── datasets                # Data loading and preprocessing scripts
│   ├── deployment              # Docker and deployment-related files
│   │   ├── api                 # FastAPI implementation
│   │   └── app                 # Streamlit application
│   └── models                  # Model training and saving scripts
├── data                        # Directory where datasets are saved when you run load_data code
└── models                      # Directory where trained model is saved
```

## Requirements

To run this project with Docker, you need:

- Python 3.9+
- Docker
- Docker Compose

## Getting Started

### Running with Docker Compose

1. **Clone the Repository**:

    ```bash
    git clone git@github.com:kkkaterinaaa/Image-classifier.git
    cd Image-classifier
    ```

2. **Build and Run the Containers**:

    In the `code/deployment` folder, run:

    ```bash
    docker-compose up --build
    ```

3. **Access the Applications**:
    - FastAPI API: [http://localhost:8000](http://localhost:8000)
    - Streamlit Web App: [http://localhost:8501](http://localhost:8501)
