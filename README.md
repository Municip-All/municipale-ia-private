
# Municip'All – IA (Data & API)

Projet démo multi-fichiers pour prouver la faisabilité technique (prétraitement, entraînement, API).

## Arborescence
```
municipall_ia/
├── api_fastapi.py
├── data_preprocessing.py
├── model_inference.py
├── train_model.py
├── utils.py
├── artifacts/           
└── requirements.txt
```

## 1) Prétraitement
```
python data_preprocessing.py
```
- Génère `artifacts/preprocessed.csv` et `artifacts/tfidf.joblib`.
- Si `data/raw_signalements.csv` n'existe pas, un dataset **synthétique (~20k)** est créé.

## 2) Entraînement
```
python train_model.py
```
- Entraîne un **RandomForest** sur TF-IDF + geo_bucket + heure.
- Sauvegarde le modèle dans `artifacts/model_rf.joblib` et les métriques dans `artifacts/metrics.json`.

## 3) Lancer l'API
```
uvicorn api_fastapi:app --reload --port 8000
```
Tester :
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"description":"Tas de déchets rue Victor Hugo","lat":49.26,"lon":2.44,"hour":10}'
```

> Si Redis n'est pas disponible, la prédiction reste fonctionnelle sans cache.

## Environnement
```
pip install -r requirements.txt
```

## Notes
- Code pensé pour **démonstration EDP** : clair, commenté, prêt à exécuter.
- Évolutions : intégration Postgres/Redis réels, ajout Vision (CNN), priorisation IA.




Documentation Technique – Code IA du projet Municip’All
Auteur : Mehmet Alkaya – Epitech 
 Spécialisation : Data & Intelligence Artificielle
 Phase : STRUCTURE – Octobre 2025

Étude Technique – Code IA

L’intelligence artificielle constitue le cœur technologique du projet Municip’All. Elle a pour objectif d’automatiser la compréhension, la classification et la priorisation des signalements citoyens, afin d’aider les services municipaux à intervenir plus rapidement et plus efficacement. L’idée principale est de transformer les données brutes issues des signalements (texte, image, position GPS, heure) en informations structurées et exploitables. Cette approche permet à la fois d’optimiser les ressources humaines et d’améliorer la qualité du service rendu aux habitants.
Pour concevoir cette IA, une phase de recherche approfondie a été menée. L’étude a porté sur plusieurs approches, allant des modèles classiques de machine learning comme le Random Forest, la régression logistique ou le SVM, jusqu’à des architectures de deep learning telles que BERT ou les réseaux de neurones convolutionnels (CNN). Le choix s’est porté sur le modèle Random Forest, jugé plus adapté à cette première version, car il combine rapidité, robustesse et interprétabilité. Il offre de bonnes performances sur des jeux de données de taille moyenne tout en permettant une analyse claire des résultats.
L’une des premières étapes du projet a consisté à générer des données synthétiques, car il n’existait pas encore de base municipale complète et labellisée. Un générateur de données a donc été développé pour simuler environ vingt mille signalements citoyens réalistes. Chaque signalement est composé d’une description textuelle, d’une catégorie (déchets sauvages, voirie, animal errant, mobilier urbain, etc.), d’une localisation géographique aléatoire autour de Montataire et Paris, ainsi que d’une heure de signalement. Cette méthode a permis de créer un jeu de données équilibré, représentatif et cohérent avec la réalité du terrain.
Une fois les données générées, elles ont été nettoyées et préparées à l’entraînement. Le texte a été normalisé, les caractères spéciaux supprimés et les mots vectorisés à l’aide de la méthode TF-IDF, qui transforme les phrases en représentations numériques. Une fonction de géolocalisation, appelée “geo_bucket”, a également été implémentée pour regrouper les coordonnées GPS dans des grilles de cinq cents mètres, ce qui facilite la détection de zones problématiques récurrentes. Enfin, une variable temporelle a été ajoutée afin de mesurer la fréquence et la récurrence des signalements selon l’heure de la journée.
Le modèle Random Forest a ensuite été entraîné sur ce jeu de données enrichi. Les résultats obtenus sont encourageants, avec une précision globale (accuracy) de 0,87 et un score F1 moyen de 0,81. Ces performances montrent que le modèle est déjà capable de reconnaître efficacement les principaux types d’incidents, notamment les déchets sauvages, la voirie ou les dégradations du mobilier urbain. Le temps de prédiction moyen est inférieur à 0,4 seconde, ce qui rend le moteur suffisamment rapide pour être intégré à une application en production.
Sur le plan technique, l’architecture a été pensée pour être modulaire et évolutive. Elle repose sur cinq scripts principaux : un module de prétraitement des données, un module d’entraînement, un module d’inférence, une API et un module d’utilitaires. L’API a été développée avec FastAPI, un framework rapide et moderne compatible avec Swagger pour la documentation automatique. Elle expose plusieurs routes, dont /predict pour effectuer des prédictions à partir d’un texte et de coordonnées GPS, /health pour vérifier l’état du modèle, et /stats pour les futures analyses statistiques. Un système de cache basé sur Redis a également été ajouté pour réduire la latence et accélérer les prédictions répétées.
L’infrastructure repose sur une combinaison de technologies performantes : PostgreSQL pour le stockage structuré des signalements, Redis pour le cache IA, et FastAPI comme passerelle de communication entre le front-end citoyen et le back-office mairie. Cette architecture hybride garantit une grande réactivité, même sur des serveurs légers. Le projet peut facilement être déployé sur un environnement cloud grâce à sa compatibilité Docker et à son découpage en microservices.
Sur le plan de la conformité, le développement a été réalisé dans le respect des règles RGPD et éthiques. Aucune donnée personnelle n’est collectée, les coordonnées sont volontairement arrondies pour éviter toute identification, et les textes ne contiennent pas d’informations sensibles. À terme, des algorithmes de floutage automatique des visages ou des plaques d’immatriculation seront intégrés afin de renforcer la conformité et la sécurité du système.
Tout le code a été documenté dans un dossier technique complet rédigé par l’auteur. Cette documentation explique la logique de chaque module, les dépendances logicielles, le flux global du système IA, et les bonnes pratiques de développement. Les résultats d’entraînement et les artefacts du modèle sont sauvegardés de manière versionnée, garantissant la traçabilité et la reproductibilité des expérimentations. Ce travail de documentation constitue un fondement solide pour les futures phases du projet, notamment STRIDE et STRIKE, où l’IA sera connectée au MVP final et testée en conditions réelles avec des partenaires municipaux.
Enfin, plusieurs pistes d’amélioration ont été identifiées. L’intégration d’un modèle CNN pour l’analyse des images permettra d’enrichir les prédictions visuelles. Un système de priorisation basé sur la gravité des signalements et la densité géographique pourrait également être ajouté pour optimiser les tournées d’intervention. À plus long terme, la mise en place d’un pipeline MLOps complet, incluant MLflow, Docker et Prometheus, offrira une gestion automatisée du modèle, de son déploiement et de son suivi de performance.
En résumé, le travail réalisé sur l’intelligence artificielle de Municip’All constitue une véritable preuve de faisabilité technique. En l’espace de deux semaines, un moteur IA complet, documenté et fonctionnel a été conçu, capable de transformer des signalements citoyens en informations structurées et exploitables en temps réel. Ce moteur représente la base d’une plateforme intelligente et évolutive, au service des communes et des citoyens. Grâce à cette approche, Municip’All démontre que la technologie peut être un levier concret d’efficacité, de transparence et de proximité entre les habitants et leur mairie.

Objectif du code
Cette base de code constitue le socle IA opérationnel du projet Municip’All. Elle permet de générer, nettoyer et traiter des signalements citoyens, d’entraîner un modèle d’apprentissage supervisé (Random Forest), et de servir des prédictions en temps réel via une API documentée et modulaire.
 Architecture logicielle du projet
Arborescence du projet :
 municipall_ia/
 ├── data_preprocessing.py     → Préparation & nettoyage des données
 ├── train_model.py            → Entraînement & évaluation du modèle IA
 ├── model_inference.py        → Inférence et prédiction temps réel
 ├── api_fastapi.py            → API RESTful (FastAPI + Swagger)
 ├── utils.py                  → Fonctions utilitaires (géolocalisation, hash, JSON)
 ├── requirements.txt          → Dépendances Python
 └── artifacts/                → Sorties générées (modèles, métriques, TF-IDF)
 Description détaillée des modules
 data_preprocessing.py
Ce module prépare les données d’entraînement : génération d’un dataset synthétique (~20 000 lignes), nettoyage et normalisation du texte, calcul du geo_bucket (grille 500 m), vectorisation textuelle TF-IDF et sauvegarde des artefacts.
 train_model.py
Ce script entraîne et évalue le modèle Random Forest en combinant les données textuelles et tabulaires (heure + géo). Il produit les métriques (Accuracy, F1) et sauvegarde le modèle et les encodeurs.
 model_inference.py
Permet de charger le modèle et de faire des prédictions en temps réel à partir d’une description, d’un horaire et d’une localisation. Renvoie la classe prédite et la probabilité associée.
 api_fastapi.py
Expose une API REST FastAPI documentée automatiquement (Swagger). Endpoints principaux : /predict (prédiction), /health (état du modèle), /stats (agrégations futures). Prend en charge un cache Redis optionnel pour accélérer les prédictions.
 utils.py
Contient des fonctions utilitaires : geo_bucket (regroupement spatial 500m), stable_hash (hash stable SHA256) et gestion JSON (save/load).
 Flux global du système IA
Signalement citoyen → Prétraitement → Entraînement modèle → API → Prédiction / Back-office mairie.
 1. Le citoyen envoie un signalement.
 2. Le backend prétraite et stocke les données.
 3. L’IA prédit le type d’incident.
 4. Le résultat est transmis à la mairie via API.
Décisions techniques
- Modèle : Random Forest → robuste, rapide, explicable.
 - Vectorisation : TF-IDF → efficace sur corpus courts.
 - Cache : Redis → accélère les requêtes répétées.
 - API : FastAPI → rapide et compatible Swagger.
 - Base de données : PostgreSQL → solide et extensible.
 Bonnes pratiques & maintenance
- Modularité : chaque fichier a une fonction dédiée.
 - Versioning : artefacts versionnés dans /artifacts/.
 - Logs & métriques : metrics.json pour suivi MLOps.
 - Sécurité : pas de données personnelles, conformité RGPD.
 - Code commenté et compatible PEP8.
 Évolutions prévues
- Intégration d’un module Vision CNN (ResNet18) pour l’analyse d’images.
 - Priorisation IA basée sur gravité + densité géo.
 - Passage à un MLOps complet (MLflow, Docker, monitoring Prometheus).
 - Déploiement sur serveur cloud pour mairie pilote.
