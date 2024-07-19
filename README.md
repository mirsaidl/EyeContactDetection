# Eye Contact Detection

This project implements a system for detecting eye contact using computer vision techniques.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The Eye Contact Detection project leverages deep learning models to detect eye contact in images and videos. It is based on the YOLOv8 model and utilizes CUDA for accelerated processing.

## Installation
To set up the environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/mirsaidl/EyeContactDetection.git
    cd EyeContactDetection
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the eye contact detection system, use the `main.py` script:

```bash
python main.py
```

Ensure you have the necessary input files and model weights in the correct directories.

## Project Structure
The repository contains the following key files and directories:

- `CUDA/`: Contains CUDA-related files for accelerated computation.
- `pics/`: Directory for storing sample images.
- `train/`: Directory for training data.
- `Yolov8Model.py`: Script for the YOLOv8 model implementation.
- `data.xlsx`: Excel file containing data.
- `main.py`: Main script to run the eye contact detection.
- `requirements.txt`: List of required Python packages.
- `utils.py`: Utility functions used across the project.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.
