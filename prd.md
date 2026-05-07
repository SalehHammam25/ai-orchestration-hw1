# PRD - AI Orchestration HW1

## 1. Project Overview

This project is HW1 for the AI Agents Orchestration course.

The goal is to build a small deep learning experiment for time-series signal reconstruction.  
The project will generate synthetic sine/cosine signals, add noise to them, and train neural networks to reconstruct the clean signal from noisy samples.

The project will compare three neural network architectures:

1. Fully Connected Neural Network / MLP
2. RNN
3. LSTM

The final project will be submitted through a public GitHub repository and documented with a detailed README file.

---

## 2. Learning Goal

The purpose of this homework is not only to produce working code, but also to practice an organized AI-assisted development workflow.

The workflow should include:

1. Writing requirements before coding.
2. Creating an implementation plan.
3. Creating a task list.
4. Implementing the project step by step.
5. Running the code.
6. Running unit tests.
7. Documenting the work clearly.
8. Pushing the work to GitHub with meaningful commits.

---

## 3. Signal Generation Requirements

The dataset should be synthetic and generated in code.

Each clean signal should be based on sine/cosine functions.

A basic sine signal can be represented as:

```text
y(t) = A * sin(2πft + φ)