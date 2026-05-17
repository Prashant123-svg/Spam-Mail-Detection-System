# Spam-Mail-Detection-System

A Deep Learning based Spam Mail Detection System built using TensorFlow, GRU (Gated Recurrent Unit), and Streamlit.

This project classifies emails/messages as Spam or Not Spam using Natural Language Processing (NLP) and Deep Learning.

---

## Features

* Spam email/message classification
* GRU-based Deep Learning model
* Text preprocessing using NLP
* Streamlit web application UI
* Tokenization and sequence padding
* Real-time spam prediction

---

## Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* NLTK

---

## Project Structure

```text
Spam-Mail-Detection-System/
│
├── app.py
├── tokenizer.pkl
├── gru_model.keras
├── gru_model_architecture.json
├── label_mapping.pkl
├── config.pkl
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Prashant123-svg/Spam-Mail-Detection-System.git
```

### Open Project Folder

```bash
cd Spam-Mail-Detection-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

## Model Architecture

The model uses:

* Embedding Layer
* GRU Layer
* Dense Output Layer

### Parameters

* Max Features: 5000
* Embedding Dimension: 32
* Max Sequence Length: 500
* GRU Units: 64

---

## Dataset

The dataset contains spam and ham (non-spam) email/messages.

Text preprocessing includes:

* Lowercasing
* Removing punctuation
* URL replacement
* Tokenization
* Stopword removal
* Sequence padding

---

## Example Predictions

### Spam Example

```text
Congratulations! You won a free iPhone. Click here now to claim your reward.
```

### Non-Spam Example

```text
Hi Prashant, your project meeting is scheduled tomorrow at 11 AM.
```

---

## Live Demo

🚀 Streamlit App:

[https://prashant123-svg-spam-mail-detection-system-app-gxvdgs.streamlit.app/](https://prashant123-svg-spam-mail-detection-system-app-gxvdgs.streamlit.app/)

---

## Deployment

This project can be deployed using:

* Streamlit Community Cloud
* Render
* Hugging Face Spaces

---

## Requirements

```text
streamlit
tensorflow-cpu==2.15.0
numpy
pandas
scikit-learn
nltk
```

---

## Author

Prashant Kumar

GitHub: [https://github.com/Prashant123-svg](https://github.com/Prashant123-svg)

---

## License

This project is open-source and available under the MIT License.

## screenshots

<img width="1907" height="917" alt="image" src="https://github.com/user-attachments/assets/bde3c315-dedc-4a69-9f7b-196c8a689b7d" />
