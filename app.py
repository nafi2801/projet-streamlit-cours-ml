import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- Configuration de la page ---
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🏦",
    layout="wide",
)

# --- Chargement du modèle, du scaler et des encodeurs (mis en cache) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoders = joblib.load("model/encoders.pkl")
    target_encoder = joblib.load("model/target_encoder.pkl")
    features = joblib.load("model/features.pkl")
    return model, scaler, encoders, target_encoder, features


@st.cache_data
def load_reference_data():
    df = pd.read_csv("data/transactions.csv", sep=";")
    return df


model, scaler, encoders, target_encoder, FEATURES = load_artifacts()
ref_df = load_reference_data()

TYPES_TRANSACTION = sorted(ref_df["Type de transaction"].unique())
STATUTS = sorted(ref_df["Status operation"].unique())
LOCALISATIONS = sorted(ref_df["Localisation"].unique())

# Fréquence moyenne d'opérations par client (pour un client inconnu, on prend la médiane)
freq_par_client = ref_df.groupby("ID Clients")["Identifiant operation"].count()
FREQ_MEDIANE = int(freq_par_client.median())

# --- En-tête ---
st.title("🏦 Système de Détection de Fraude Bancaire")
st.markdown(
    "Analysez une transaction ou un lot de transactions pour détecter un risque de fraude "
    "(3 classes : **Normal**, **Suspect**, **Fraude**)."
)

# --- Menu latéral ---
mode = st.sidebar.radio("Mode d'analyse", ["Transaction unique", "Fichier CSV (lot)"])


def build_features(montant, date_heure, type_transaction, statut, localisation, id_client):
    """Construit le vecteur de features attendu par le modèle à partir des champs saisis."""
    heure = date_heure.hour
    jour_semaine = date_heure.weekday()
    nuit = 1 if (heure >= 22 or heure <= 5) else 0

    nb_operations_client = freq_par_client.get(id_client, FREQ_MEDIANE)

    loc_client_hist = ref_df[ref_df["ID Clients"] == id_client]["Localisation"]
    loc_frequente_client = int((loc_client_hist == localisation).sum() >= 2)

    def safe_encode(col, value):
        le = encoders[col]
        if value in le.classes_:
            return le.transform([value])[0]
        # valeur inconnue -> on retombe sur la classe la plus fréquente
        return le.transform([le.classes_[0]])[0]

    row = {
        "Montant": montant,
        "Heure": heure,
        "Jour_semaine": jour_semaine,
        "Nuit": nuit,
        "Nb_operations_client": nb_operations_client,
        "Loc_frequente_client": loc_frequente_client,
        "Type de transaction_enc": safe_encode("Type de transaction", type_transaction),
        "Status operation_enc": safe_encode("Status operation", statut),
        "Localisation_enc": safe_encode("Localisation", localisation),
    }
    return pd.DataFrame([row])[FEATURES]


def label_style(label):
    if label == "Fraude":
        return "background-color: #ffb3b3"
    if label == "Suspect":
        return "background-color: #ffe0b3"
    return ""


# ============================
# MODE 1 : Transaction unique
# ============================
if mode == "Transaction unique":
    st.subheader("Saisie manuelle d'une transaction")

    col1, col2 = st.columns(2)
    with col1:
        id_client = st.number_input("ID Client", min_value=1, value=int(ref_df["ID Clients"].iloc[0]), step=1)
        montant = st.number_input("Montant de la transaction (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
        date_val = st.date_input("Date de la transaction", value=datetime.now().date())
        heure_val = st.slider("Heure de la transaction (0-23)", 0, 23, 12)

    with col2:
        type_transaction = st.selectbox("Type de transaction", TYPES_TRANSACTION)
        statut = st.selectbox("Statut de l'opération", STATUTS)
        localisation = st.selectbox("Localisation", LOCALISATIONS)

    if st.button("Analyser la transaction", type="primary"):
        date_heure = datetime.combine(date_val, datetime.min.time()).replace(hour=heure_val)
        X = build_features(montant, date_heure, type_transaction, statut, localisation, id_client)
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        label = target_encoder.inverse_transform([prediction])[0]
        proba_dict = dict(zip(target_encoder.classes_, proba))

        st.divider()

        if label == "Fraude":
            st.error(f"🚨 **Transaction frauduleuse** — Probabilité : {proba_dict['Fraude']:.1%}")
        elif label == "Suspect":
            st.warning(f"⚠️ **Transaction suspecte** — Probabilité : {proba_dict['Suspect']:.1%}")
        else:
            st.success(f"✅ **Transaction normale** — Probabilité : {proba_dict['Normal']:.1%}")

        st.markdown("**Répartition des probabilités par classe**")
        proba_df = pd.DataFrame({
            "Classe": list(proba_dict.keys()),
            "Probabilité": list(proba_dict.values()),
        }).sort_values("Probabilité", ascending=False)
        st.bar_chart(proba_df.set_index("Classe"))

# ============================
# MODE 2 : Fichier CSV (lot)
# ============================
else:
    st.subheader("Analyse par lot (fichier CSV)")
    st.caption(
        "Colonnes attendues : ID Clients ; Numero de compte ; Identifiant operation ; "
        "Type de transaction ; Status operation ; Localisation ; Date ; Montant"
    )

    fichier = st.file_uploader("Déposez un fichier CSV de transactions (séparateur ';')", type=["csv"])

    if fichier is not None:
        df_batch = pd.read_csv(fichier, sep=";")
        st.write("Aperçu des données :", df_batch.head())

        if st.button("Lancer l'analyse du lot"):
            df_batch["Date"] = pd.to_datetime(df_batch["Date"])

            rows = []
            for _, r in df_batch.iterrows():
                rows.append(
                    build_features(
                        r["Montant"], r["Date"], r["Type de transaction"],
                        r["Status operation"], r["Localisation"], r["ID Clients"],
                    ).iloc[0]
                )
            X_batch = pd.DataFrame(rows)[FEATURES]
            X_batch_scaled = scaler.transform(X_batch)

            preds = model.predict(X_batch_scaled)
            probas = model.predict_proba(X_batch_scaled)

            df_batch["prediction"] = target_encoder.inverse_transform(preds)
            df_batch["probabilite_fraude"] = probas[:, list(target_encoder.classes_).index("Fraude")]
            df_batch["probabilite_suspect"] = probas[:, list(target_encoder.classes_).index("Suspect")]

            nb_fraudes = (df_batch["prediction"] == "Fraude").sum()
            nb_suspects = (df_batch["prediction"] == "Suspect").sum()
            st.warning(
                f"**{nb_fraudes}** transaction(s) frauduleuse(s) et **{nb_suspects}** "
                f"suspecte(s) détectée(s) sur {len(df_batch)}."
            )

            st.dataframe(
                df_batch.style.apply(
                    lambda row: [label_style(row["prediction"]) for _ in row],
                    axis=1,
                )
            )

            csv_export = df_batch.to_csv(index=False, sep=";").encode("utf-8")
            st.download_button(
                "Télécharger les résultats", csv_export, "resultats_analyse.csv", "text/csv"
            )

# --- Pied de page ---
st.sidebar.markdown("---")
st.sidebar.caption("Projet pédagogique — Détection de fraude bancaire par IA")
