# Campus Lost-and-Found: A Multi-Stage Neural Network System

**ECE 539 Fall 2026 | Team G21**

## Overview
This repository contains a multi-stage neural network system designed to improve matching between lost and found item reports compared to traditional keyword search. The pipeline accepts a lost-item description, approximate date, and campus location, returning up to three ranked found-item reports. 

The architecture integrates neural classification, semantic matching, and metadata filtering using:
* **DistilBERT:** Item classification
* **Sentence-BERT (MiniLM):** Dense embeddings for cosine similarity matching
* **BM25:** Lexical baseline comparison
* **Rule-Based Logic:** Metadata filtering (date/location) and final abstention thresholding

**Data Source:** Drishya Tomar’s LostFound Kaggle dataset.

## Team Members and Responsibilities
* **Rob Calvo** - Workspace `P1`: Rule-based final ranking and evaluation metrics scripts (Recall@3, MRR, Macro-F1).
* **Parith Reddy** - Workspace `P2`: Implement DistilBERT item classifier and baseline BM25 comparison.
* **Vihan Vivek Dalvi** - Workspace `P3`: Implement Sentence-BERT Semantic Matcher and end-to-end PyTorch pipeline integration.
* **Philip Mann-Segal** - Workspace `P4`: Dataset preprocessing, category label mapping, and Candidate Filter logic.

## Repository Structure

Development Timeline: P4 -> P2 -> P3 -> P1
System Run Pipeline: P2 -> P4 -> P3 -> P1

* `/P1/` - Final ranking and evaluation logic
* `/P2/` - BM25 baseline and DistilBERT classification
* `/P3/` - Semantic matching and end-to-end pipeline integration
* `/P4/` - Data preprocessing and candidate filtering
* `/final/` - Integrated project deliverable

## Tech Stack
* Python, PyTorch, Hugging Face Transformers, Sentence Transformers, scikit-learn