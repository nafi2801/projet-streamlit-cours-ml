"""
Entraînement du modèle de détection de fraude bancaire
Dataset : transactions.csv (ID Clients, Numero de compte, Identifiant operation,
          Type de transaction, Status operation, Localisation, Date, Montant, Target)

Target à 3 classes : Normal / Suspect / Fraude
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. Chargement des données
# ------------------------------------------------------------------
df = pd.read_csv("data/transactions.csv", sep=";")

# ------------------------------------------------------------------
# 2. Feature engineering
# ------------------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"])
df["Heure"] = df["Date"].dt.hour
df["Jour_semaine"] = df["Date"].dt.dayofweek          # 0 = lundi
df["Nuit"] = df["Heure"].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)

# Fréquence historique de l'opération par client (proxy de comportement habituel)
freq_client = df.groupby("ID Clients")["Identifiant operation"].transform("count")
df["Nb_operations_client"] = freq_client

# Fréquence par localisation (une localisation rare pour un client peut être un signal)
loc_par_client = df.groupby(["ID Clients", "Localisation"])["Localisation"].transform("count")
df["Loc_frequente_client"] = (loc_par_client >= 2).astype(int)

# ------------------------------------------------------------------
# 3. Encodage des variables catégorielles
# ------------------------------------------------------------------
cat_cols = ["Type de transaction", "Status operation", "Localisation"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

target_encoder = LabelEncoder()
df["Target_enc"] = target_encoder.fit_transform(df["Target"])  # Fraude / Normal / Suspect

# ------------------------------------------------------------------
# 4. Sélection des features / cible
# ------------------------------------------------------------------
FEATURES = [
    "Montant",
    "Heure",
    "Jour_semaine",
    "Nuit",
    "Nb_operations_client",
    "Loc_frequente_client",
    "Type de transaction_enc",
    "Status operation_enc",
    "Localisation_enc",
]

X = df[FEATURES]
y = df["Target_enc"]

# ------------------------------------------------------------------
# 5. Normalisation
# ------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------
# 6. Split train/test
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------------------------------------------
# 7. Entraînement
#    Point d'attention : classes fortement déséquilibrées
#    (Normal >> Suspect >> Fraude) -> class_weight="balanced"
# ------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 8. Évaluation
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
print("Classes :", list(target_encoder.classes_))
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nImportance des variables :")
print(importances)

# ------------------------------------------------------------------
# 9. Sauvegarde du modèle, du scaler et des encodeurs
# ------------------------------------------------------------------
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(target_encoder, "model/target_encoder.pkl")
joblib.dump(FEATURES, "model/features.pkl")

print("\nModèle, scaler et encodeurs sauvegardés dans model/")
