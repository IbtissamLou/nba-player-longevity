# 🏀 NBA Player Longevity Predictor

**📌 Description du Projet**

Ce projet a pour objectif de prédire si un joueur NBA aura une carrière de plus de 5 ans, en utilisant des modèles de classification basés sur le Machine Learning. L'API REST développée avec FastAPI permet d'envoyer les statistiques d'un joueur et d'obtenir une prédiction en retour.

**🚀 Fonctionnalités Principales**

Prédiction de la longévité d'un joueur NBA (Carrière > 5 ans ou non).

- Modèles de classification utilisés :<br>
Random Forest (RF)<br>
Balanced Random Forest (BalancedRF)<br>
XGBoost Classifier<br>

- Optimisation des hyperparamètres avec Optuna.
- Évaluation des modèles basée sur le F1-score et l'équilibre entre Précision & Recall.
- API REST pour requêtes unitaires .
- Interface web avec formulaire et rendu dynamique via Jinja2.


**⚙️ Installation et Exécution Locale**

1. Cloner le dépôt
   
git clone https://github.com/IbtissamLou/nba-player-longevity.git
cd nba-player-longevity

2. Créer un environnement virtuel
   
python -m venv env
source env/bin/activate

3. Installer les dépendances
   
pip install -r requirements.txt

4. Lancer l’API FastAPI
   
uvicorn app.main:app --reload


➡️ L’API sera disponible sur :
👉 http://127.0.0.1:8000

**🎯 Utilisation de l’API**

🔹 1. Interface Web

Accédez à :
👉 http://127.0.0.1:8000

Remplissez les statistiques du joueur et cliquez sur Predict pour obtenir le résultat.

🔹 2. Requête API (via cURL ou Postman)

curl -X 'POST' 'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'GP=72&MIN=30.2&PTS=15.3&FGM=5.8&FGA=12.7&FG_Percentage=45.8&ThreeP_Made=1.2&ThreePA=3.6&ThreeP_Percentage=33.3&FTM=3.2&FTA=4.0&FT_Percentage=80.2&OREB=1.1&DREB=4.5&REB=5.6&AST=4.3&STL=1.2&BLK=0.5&TOV=2.1'

**🧪 🔁 Continuous Testing & Integration**

Tests unitaires automatisés avec pytest

Test de l’API FastAPI et du modèle ML

Intégration continue via GitHub Actions (CI)

Chaque push sur la branche principale déclenche les tests

✅ Si tous les tests passent → la build est validée

![CI](https://github.com/IbtissamLou/nba-player-longevity/actions/workflows/ci.yml/badge.svg)

**🚚 🚀 Continuous Delivery**

Une pipeline CD (GitHub Actions) automatise :

le build Docker multi-plateforme (linux/amd64)

le push automatique vers Docker Hub :
ibti2/nba-prediction-api

docker pull --platform linux/amd64 ibti2/nba-prediction-api:latest
docker run --platform linux/amd64 -p 8000:8000 ibti2/nba-prediction-api:latest

![CD](https://github.com/IbtissamLou/nba-player-longevity/actions/workflows/cd.yml/badge.svg)


**🔥 Améliorations Futures**

🚀 Continuous Deployment (CDP) vers un serveur cloud (AWS / GCP / Azure)

🤖 Intégration de modèles plus complexes (Deep Learning si plus de données)

🧩 Monitoring en production (Prometheus / Grafana)

🎨 Interface plus intuitive (Streamlit ou React front-end)

**🧑‍💻 Auteurs**
LOUKILI Ibtissam - Contact : ibtissamloukili20@gmail.com


