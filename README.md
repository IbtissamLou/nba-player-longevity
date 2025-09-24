# 🏀 NBA Player Longevity Predictor

**📌 Description du Projet**

Ce projet a pour objectif de prédire si un joueur NBA aura une carrière de plus de 5 ans, en utilisant des modèles de classification basés sur le Machine Learning. L'API REST développée avec FastAPI permet d'envoyer les statistiques d'un joueur et d'obtenir une prédiction en retour.

**🚀 Fonctionnalités Principales**

Prédiction de la longévité d'un joueur NBA (Carrière > 5 ans ou non).

- Modèles de classification utilisés :
Random Forest (RF)
Balanced Random Forest (BalancedRF)
XGBoost Classifier

- Optimisation des hyperparamètres avec Optuna.
- Évaluation des modèles basée sur le F1-score et l'équilibre entre Précision & Recall.
- API REST pour requêtes unitaires .
- Interface utilisateur permettant de saisir les statistiques d’un joueur et d’afficher la prédiction.


**📂 Structure du Projet**

NBA_Career_Prediction/
│── rf_model/                        # Modèles entraînés sauvegardés en .pkl
│── templates/                     # Fichiers HTML pour l'interface utilisateur
│   ├── index.html                 # Formulaire de saisie des données
│   ├── result.html                 # Page de résultats
│── static/                          # Dossier pour fichiers CSS
|── Data_prep.ipynb                   #Script préparation des données
│── Model_Intég.py                   # Code principal de l'API REST
│── Model_classif.ipynb              # Script d'entraînement des modèles
│── requirements.txt                 # Bibliothèques nécessaires
│── README.md                        # Documentation du projet
│── scoring_kfold.py                 # Fonction scoring basic
│── scoring_optim.py                 # Fonction scoring avec optimisation GridSearch
│── scoring_optuna.py                # Fonction scoring avec optimisatiton Optuna 


**📦 Installation et Dépendances**

- Créer un environnement virtuel et l’activer
python -m venv env
source env/bin/activate  
- Installer les dépendances
pip install -r requirements.txt

**🎯 Utilisation de l'API REST**

📌 1. Lancer l'API
Démarrez le serveur FastAPI avec Uvicorn :

uvicorn Model_depy:app --reload
L'API sera accessible à l'adresse http://127.0.0.1:8000

📌 2. Tester l'API via cURL
Vous pouvez envoyer une requête POST contenant les statistiques d'un joueur :

curl -X 'POST' 'http://127.0.0.1:8000/predict' \
-H 'Content-Type: application/json' \
-d '{
    "GP": 72,
    "MIN": 30.2,
    "PTS": 15.3,
    "FGM": 5.8,
    "FGA": 12.7,
    "FG_Percentage": 45.8,
    "ThreeP_Made": 1.2,
    "ThreePA": 3.6,
    "ThreeP_Percentage": 33.3,
    "FTM": 3.2,
    "FTA": 4.0,
    "FT_Percentage": 80.2,
    "OREB": 1.1,
    "DREB": 4.5,
    "REB": 5.6,
    "AST": 4.3,
    "STL": 1.2,
    "BLK": 0.5,
    "TOV": 2.1
}'

📌 3. Tester l'API via l'interface web
Ouvrez votre navigateur et accédez à :

http://127.0.0.1:8000
Remplissez le formulaire avec les statistiques du joueur et cliquez sur Predict pour voir le résultat.


**🔥 Améliorations Futures**

📌 Intégration de nouveaux modèles (Deep Learning si plus de données disponible?)
📊 Amélioration des features sélectionnées
🚀 Déploiement sur un serveur cloud
🎨 Interface plus intuitive avec Streamlit

**🧑‍💻 Auteurs**
LOUKILI Ibtissam - Contatc : ibtissamloukili20@gmail.com



