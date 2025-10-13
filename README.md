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
- Interface utilisateur permettant de saisir les statistiques d’un joueur et d’afficher la prédiction.


**📂 Structure du Projet**

NBA_Career_Prediction/  
│── rf_model/                        # Modèles entraînés sauvegardés en .pkl<br>
│── templates/                     # Fichiers HTML pour l'interface utilisateur<br>
│   ├── index.html                 # Formulaire de saisie des données<br>
│   ├── result.html                 # Page de résultats<br>
│── static/                          # Dossier pour fichiers CSS<br>
|── Data_prep.ipynb                   #Script préparation des données<br>
│── Model_Intég.py                   # Code principal de l'API REST<br>
│── Model_classif.ipynb              # Script d'entraînement des modèles<br>
│── requirements.txt                 # Bibliothèques nécessaires<br>
│── README.md                        # Documentation du projet<br>
│── scoring_kfold.py                 # Fonction scoring basic<br>
│── scoring_optim.py                 # Fonction scoring avec optimisation GridSearch<br>
│── scoring_optuna.py                # Fonction scoring avec optimisatiton Optuna<br> 


**📦 Installation et Dépendances**

- Créer un environnement virtuel et l’activer<br>
python -m venv env<br>
source env/bin/activate  <br>
- Installer les dépendances<br>
pip install -r requirements.txt<br>

**🎯 Utilisation de l'API REST**

1. Lancer l'API<br>
Démarrez le serveur FastAPI avec Uvicorn :<br>

uvicorn Model_depy:app --reload<br>
L'API sera accessible à l'adresse http://127.0.0.1:8000<br>

2. Tester l'API via cURL<br>
Vous pouvez envoyer une requête POST contenant les statistiques d'un joueur :<br>

curl -X 'POST' 'http://127.0.0.1:8000/predict' \
-H 'Content-Type: application/json' \
-d '{<br>
    "GP": 72,<br>
    "MIN": 30.2,<br>
    "PTS": 15.3,<br>
    "FGM": 5.8,<br>
    "FGA": 12.7,<br>
    "FG_Percentage": 45.8,<br>
    "ThreeP_Made": 1.2,<br>
    "ThreePA": 3.6,<br>
    "ThreeP_Percentage": 33.3,<br>
    "FTM": 3.2,<br>
    "FTA": 4.0,<br>
    "FT_Percentage": 80.2,<br>
    "OREB": 1.1,<br>
    "DREB": 4.5,<br>
    "REB": 5.6,<br>
    "AST": 4.3,<br>
    "STL": 1.2,<br>
    "BLK": 0.5,<br>
    "TOV": 2.1<br>
}'<br>

3. Tester l'API via l'interface web<br>
Ouvrez votre navigateur et accédez à :<br>

http://127.0.0.1:8000<br>
Remplissez le formulaire avec les statistiques du joueur et cliquez sur Predict pour voir le résultat.

![CI](https://github.com/IbtissamLou/nba-player-longevity/actions/workflows/ci.yml/badge.svg)

**🔥 Améliorations Futures**

- Intégration de nouveaux modèles (Deep Learning si plus de données disponible?)
- Amélioration des features sélectionnées
- Déploiement sur un serveur cloud
- Interface plus intuitive avec Streamlit

**🧑‍💻 Auteurs**
LOUKILI Ibtissam - Contact : ibtissamloukili20@gmail.com


