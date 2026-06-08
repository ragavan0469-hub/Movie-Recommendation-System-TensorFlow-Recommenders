# Deep Learning Movie Recommendation System

## Overview

This project implements a Deep Learning Movie Recommendation System using TensorFlow Recommenders (TFRS) and a Two-Tower Neural Network Architecture.

The system learns user preferences and movie representations through embedding layers and generates personalized movie recommendations.

## Features

- TensorFlow Recommenders
- Two-Tower Architecture
- User Embeddings
- Movie Embeddings
- Personalized Recommendations
- Top-K Retrieval
- Deep Learning Based Recommendation

## Dataset

Movies Dataset:
- movieId
- title
- genres

Ratings Dataset:
- userId
- movieId
- rating
- timestamp

Combined Dataset:
- userId
- movieId
- rating
- timestamp
- title
- genres

## Technologies

- Python
- TensorFlow
- TensorFlow Recommenders
- Pandas
- NumPy
- Google Colab

## Model Architecture

User Tower
→ User Embedding

Movie Tower
→ Movie Embedding

Dot Product Similarity

Top-K Recommendations

## Results

The model generates personalized movie recommendations based on user interaction history.

## Run

python train.py

python recommend.py
