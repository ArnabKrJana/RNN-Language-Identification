<div align="center">

# 🌍 Language Detection using RNN

### 🔍 Predict the language of any text using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

<p align="center">
  <em>
    This project demonstrates a complete end-to-end machine learning workflow,<br> 
    including data analysis, model training, and deployment.
  </em>
</p>

</div>

---

## 📌 Project Description

Language detection is a fundamental Natural Language Processing (NLP) task used in applications such as **multilingual chat systems**, **search engines**, and **translation pipelines**.

In this project, a **Simple RNN (Recurrent Neural Network)** is trained to classify text into its corresponding language based on learned textual patterns. The project is divided into three major components:
1.  **Exploratory Data Analysis & Preprocessing**
2.  **Model Training & Prediction Pipeline**
3.  **Streamlit Web Application**

---

## 🧠 Workflow Overview

### 1. Exploratory Data Analysis & Preprocessing
* Performed analysis on multilingual text data.
* Cleaned and prepared text for sequence modeling.
* Converted text into numerical sequences using a tokenizer.
* Encoded language labels using label encoding.
* **Relevant File:** `eda.ipynb`

### 2. Model Building & Training
* Built a **Simple RNN-based neural network** using TensorFlow & Keras.
* Used padded sequences for uniform input length.
* Trained the model to learn sequential language patterns.
* **Artifacts Saved:**
    * `simple_rnn_model.h5` (Trained Model)
    * `tokenizer.pkl` (Tokenizer & Label Encoder)
* **Relevant File:** `prediction.ipynb`

### 3. Prediction Pipeline
* Loads the trained RNN model and tokenizer.
* Converts user input text into padded sequences.
* Predicts the language and provides a confidence score.
* **Relevant File:** `app.py`

---

## 🌐 Streamlit Web Application

An interactive web app allowing users to enter text and get instant predictions.

**Key Features:**
* ✨ Clean User Interface
* ⚡ Cached model loading for efficiency
* 🚀 Real-time inference

### Example Usage
| Input | Output |
| :--- | :--- |
| **User Text:** `यह एक अच्छा दिन है` | **Prediction:** Hindi <br> **Confidence:** 98.5% |

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Programming & ML** | Python, TensorFlow, Keras |
| **NLP & Data** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit |
| **Utilities** | Pickle, IPyKernel |

---

## 📦 Installation & Setup

Follow these steps to run the project locally.

**Step 1: Clone the Repository**
```bash
git clone https://github.com/ArnabKrJana/RNN-Language-Identification.git
cd language-detection-rnn
