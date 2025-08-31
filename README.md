# DermaScope AI

An AI-powered web application for dermatology screening, featuring dual models for public screening of common skin conditions and professional-grade skin cancer detection.

## Features

* **Public Screening Model:** Classifies 9 different non-cancerous skin conditions (e.g., Eczema, Psoriasis).
* **Cancer Detection Model:** Classifies 4 categories, including Melanoma and Basal-cell carcinoma, for professional use.
* **Web Interface:** Simple and intuitive frontend built with HTML, CSS, and JavaScript.
* **AI Backend:** A powerful Flask and PyTorch server that serves the deep learning models.

## Tech Stack

* **Backend:** Python, Flask, PyTorch
* **Frontend:** HTML, CSS, JavaScript
* **Deployment:** Render (Backend), Netlify (Frontend)

## How to Run Locally

1.  Clone the repository.
2.  Create and activate a virtual environment: `python3 -m venv venv` and `source venv/bin/activate`.
3.  Install dependencies: `pip install -r requirements.txt`.
4.  Run the Flask server: `python app.py`.