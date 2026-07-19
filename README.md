# Détection de fraude bancaire — Streamlit

Application de détection de fraude entraînée sur `data/transactions.csv`
(5 382 transactions, colonnes : ID Clients, Numero de compte, Identifiant
operation, Type de transaction, Status operation, Localisation, Date, Montant,
Target). La cible `Target` a 3 classes : **Normal**, **Suspect**, **Fraude**.

## Structure du projet

```
fraude-detection-streamlit/
├── data/
│   └── transactions.csv          # jeu de données
├── model/
│   ├── train_model.py            # script d'entraînement
│   ├── fraud_model.pkl           # modèle RandomForest sérialisé
│   ├── scaler.pkl                # StandardScaler
│   ├── encoders.pkl              # LabelEncoders des variables catégorielles
│   ├── target_encoder.pkl        # LabelEncoder de la cible
│   └── features.pkl              # ordre des features attendu par le modèle
├── app.py                        # application Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Entraîner le modèle

```bash
python model/train_model.py
```

Résultats obtenus sur le jeu de test (20 %, stratifié) :

| Classe  | Précision | Rappel | F1-score | Support |
|---------|-----------|--------|----------|---------|
| Fraude  | 0.89      | 0.78   | 0.83     | 40      |
| Normal  | 0.95      | 0.96   | 0.95     | 819     |
| Suspect | 0.84      | 0.81   | 0.83     | 218     |
| **Accuracy globale** | | | **0.92** | 1077 |

Variables les plus discriminantes : `Montant`, `Nb_operations_client`
(fréquence historique du client), `Status operation`, `Localisation`.

## Lancer l'application

```bash
streamlit run app.py
```

Deux modes :
- **Transaction unique** — saisie manuelle (client, montant, date/heure,
  type de transaction, statut, localisation) → prédiction + probabilités
  par classe.
- **Fichier CSV (lot)** — dépôt d'un CSV avec le même format que
  `data/transactions.csv` (séparateur `;`) → prédictions pour toutes les
  lignes, export des résultats.

## Déploiement sur Streamlit Community Cloud

1. Pousser le dépôt sur GitHub (le `.gitignore` exclut `venv/` et les caches).
2. Sur [share.streamlit.io](https://share.streamlit.io), se connecter avec
   GitHub, cliquer sur **New app**, choisir le dépôt, la branche `main` et
   `app.py` comme fichier principal, puis **Deploy**.
3. Pour des identifiants sensibles (BDD, API), utiliser **App settings →
   Secrets** (format TOML), jamais de valeurs codées en dur.

## Points d'attention

- Les classes sont déséquilibrées (Normal ≫ Suspect ≫ Fraude) : le modèle
  utilise `class_weight="balanced"` — privilégier precision/recall/F1 plutôt
  que l'accuracy seule pour juger la qualité.
- Aucune donnée client réelle sensible ne doit être poussée sur un dépôt
  public ; anonymiser ou synthétiser si besoin.
- Nommer les versions du modèle (`fraud_model_v1.pkl`, etc.) pour tracer les
  évolutions dans le temps.
