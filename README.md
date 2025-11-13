## Table of Contents
- [Motivation](#Motivation)
- [Project Goals](#Project_Goals)
- [Installation](#installation)
- [Usage](#Usage)
- [Contributing](#contributing)
- [License](#license)


# Motivation
This project was created as a personal learning journey to understand how Large Language Models (LLMs) are built from the ground up.
Inspired by Sebastian Raschka’s book “Build a Large Language Model (From Scratch)”, it re-implements the key concepts and architecture of GPT-2, following a step-by-step, hands-on approach.
Each module is designed to deepen understanding of transformer components, such as tokenization, attention, and training loops — making it a practical guide for anyone who wants to learn how LLMs work internally.

## Project Goals:
Each module could be executed on its own to understand how it works and check its output for better understanding.
- Understand the inner workings of transformer-based models like GPT-2
- Implement key components step-by-step (tokenization, attention, training loop, etc.)
- Learn the fundamentals of how modern LLMs are built and optimized



## Installation
- Python 3.x required
- Install dependencies:
## Clone the repository

git clone https://github.com/sunishbharat/llm-from-the-ground-up.git
cd llm-from-the-ground-up


## Install dependencies

pip install -r requirements.txt


## Run an individual module to explore

python test.py

## Usage
To run in Google colab, Main script to execute for Inference and training.
 Pretraining_llm:
 - Run entire script to evaluate the default training loop, set for 100 epochs, sufficient enough to complete in few minutes.
 - Display the training loss wrt to batches processed.
 - Performs the inference based on the training loops, tweak the epochs to see improvement in performance.
 - Hyperparameters are set in config file.
