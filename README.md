# 🏀 NBA Player Longevity Predictor

### **📌 Project Description**

This project aims to predict whether an NBA player will have a career lasting more than 5 years, using Machine Learning classification models.
A FastAPI REST API allows users to send a player’s statistics and receive a prediction in response.

### **🚀 Main Features**

Prediction of NBA player longevity (career > 5 years or not).

- Classification models used :<br>
Random Forest (RF)<br>
Balanced Random Forest (BalancedRF)<br>
XGBoost Classifier<br>

- Hyperparameter optimization using Optuna.
- Model evaluation based on F1-score and balance between Precision & Recall.
- REST API for single prediction requests.
- Web interface with a form and dynamic rendering via Jinja2.


### **⚙️ Local Installation & Execution**

1. Clone the repository
   
git clone https://github.com/IbtissamLou/nba-player-longevity.git
cd nba-player-longevity

2. Create a virtual environment
   
python -m venv env
source env/bin/activate

3. Install dependencies
   
pip install -r requirements.txt

4. Launch the FastAPI application
   
uvicorn app.main:app --reload


➡️ The API will be available at:
👉 http://127.0.0.1:8000

### **🎯 API Usage**

🔹 1. Web Interface

Go to :
👉 http://127.0.0.1:8000

Fill in the player’s statistics and click Predict to get the result.

### **🧪 🔁 Continuous Testing & Integration**

Automated unit testing with pytest

Tests for FastAPI API and ML model

Continuous Integration (CI) with GitHub Actions

Each push to the main branch triggers all tests

✅ If all tests pass → the build is validated

![CI](https://github.com/IbtissamLou/nba-player-longevity/actions/workflows/ci.yml/badge.svg)

### **🚚 🚀 Continuous Delivery**

A CD pipeline (GitHub Actions) automates:

Multi-platform Docker build (linux/amd64)

Automatic push to Docker Hub → ibti2/nba-prediction-api

docker pull --platform linux/amd64 ibti2/nba-prediction-api:latest

docker run --platform linux/amd64 -p 8000:8000 ibti2/nba-prediction-api:latest

![CD](https://github.com/IbtissamLou/nba-player-longevity/actions/workflows/cd.yml/badge.svg)


### **🚀 Déploiement Continu (CDP) — Azure App Service**

After setting up Continuous Integration (CI) and Continuous Delivery (CD),
the final step of the MLOps pipeline is Continuous Deployment (CDP) — meaning automatic deployment of every validated version to production.

In this project, deployment is handled using Azure App Service for Containers 🌐.

Each push to the main branch triggers the following automated steps:

- GitHub Actions builds the Docker image of the FastAPI application.

- The image is pushed to Docker Hub (ibti2/nba-prediction-api:latest).

- Using the Azure publish profile (stored as a GitHub secret), GitHub Actions authenticates with Azure.

- The new image is automatically deployed to the web app hosted on Azure.

👉 Result: Every code update triggers a complete pipeline — from testing to deployment — with no manual intervention 🚀

🔑 Technologies Used

GitHub Actions → workflow automation

Docker Hub → container image hosting

Azure App Service → production container hosting

### **🔐 Secure Access Management (GitHub Secrets)**

DOCKER_USERNAME and DOCKER_PASSWORD → Docker Hub authentication

AZURE_WEBAPP_PUBLISH_PROFILE → secure authentication to Azure App Service


### **🔥 Future Improvements**

🤖 Integration of more advanced models
Incorporate Deep Learning architectures (e.g., Neural Networks) when more player data becomes available.

🧩 Production Monitoring
Add real-time monitoring and performance tracking using tools like Prometheus and Grafana.

🎨 More Intuitive Interface
Develop a modern and user-friendly interface using Streamlit or a React front-end connected to the FastAPI backend.

### **🧑‍💻 Authors**
Ibtissam Lou — Data Scientist & ML Engineer - Contact : ibtissamloukili20@gmail.com
