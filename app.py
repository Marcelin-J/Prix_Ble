import streamlit as st
import pandas as pd
import numpy as np
# On retire matplotlib et seaborn
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Plus utilisé pour affichage simple

# --- Configuration ---
# Le fichier Excel doit être dans le même répertoire que ce script app.py sur GitHub
EXCEL_FILE = 'BBD_Prix_Ble.xlsx'
SHEET_NAME = 'Feuil1'
DATE_COLUMN = 'DATE'
TARGET_COLUMN = 'Prix_Ble'

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Prédiction du Prix du Blé")

# --- Zone de Texte Personnalisable ---
st.markdown("## Simulateur de Prix du Blé")
custom_text = st.text_area(
    "Ce simulateur est réalisé dans le cadre d'un projet agrosciences du cycle ingénieur par 5 étudiants",
    "Ce site utilise 11 facteurs choisis car ils présentent en théorie un impact sur la variabilité du prix du blé. La simulation est basée sur un modèle de Régression Linéaire simple en se basant sur ces facteurs historiques et en fonction de leur corrélation avec le prix du blé. Il utilse une approche itérative pour gérer le prix décalé du mois précédent.",
    " ",
    "Vous pouvez sélectionner le nombre de mois pour l'estimation future entre 1 et 12.",
    "Auteurs : Marcelin JOLLET, Geuthier DEFONTE, Elise DURLIN, Domitille CANESSON, Leo DE LA FAYE"
)
st.write(custom_text)
st.markdown("---") # Séparateur visuel

# --- Slider pour le nombre de mois de prédiction ---
n_prediction_months = st.slider(
    "Nombre de mois à prédire dans le futur :",
    min_value=1,
    max_value=12,
    value=6,
    step=1
)

# --- Chargement et Préparation des Données ---

@st.cache_data # Cache les données après le premier chargement
def load_and_preprocess_data(file_path, sheet_name, date_column, target_column):
    """Charge, nettoie et prépare les données."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        st.error(f"ERREUR : Fichier non trouvé à l'emplacement {file_path}. Assurez-vous qu'il est dans le même répertoire.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"ERREUR lors du chargement du fichier Excel : {e}")
        return None, None, None, None, None

    # Conversion et indexation de la date
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
    except Exception as e:
        st.error(f"ERREUR lors du traitement de la colonne de date '{date_column}' : {e}. Assurez-vous qu'elle existe et est au format date.")
        return None, None, None, None, None

    # Gérer les valeurs manquantes (NaN)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    if df.isnull().sum().sum() > 0:
         st.warning("Certaines valeurs manquantes subsistent après l'imputation. Vérifiez vos données sources.")

    # Séparer cible et caractéristiques
    if target_column not in df.columns:
         st.error(f"ERREUR : La colonne cible '{target_column}' n'a pas été trouvée dans le fichier.")
         st.write(f"Colonnes disponibles : {df.columns.tolist()}")
         return None, None, None, None, None

    y = df[target_column]
    # S'assurer que la colonne cible est une série
    if not isinstance(y, pd.Series):
        y = pd.Series(y.values, index=y.index, name=target_column)

    X_initial = df.drop(columns=[target_column]).select_dtypes(include=np.number)

    if X_initial.empty:
        st.warning("Aucune colonne de facteurs numériques trouvée après avoir retiré la colonne cible. Le modèle utilisera uniquement des caractéristiques temporelles et le lag du prix.")
        # Créer un DataFrame X_initial vide pour que le code puisse continuer
        X_initial = pd.DataFrame(index=df.index)

    return df, y, X_initial, date_column, target_column

df, y, X_initial, date_column_name, target_column_name = load_and_preprocess_data(EXCEL_FILE, SHEET_NAME, DATE_COLUMN, TARGET_COLUMN)

if df is not None: # Continuer seulement si les données ont été chargées et traitées sans erreur majeure

    # --- Ingénierie de Caractéristiques ---
    @st.cache_data
    def engineer_features(X_initial_df, y_series):
        """Ajoute des caractéristiques temporelles et le lag."""
        X = X_initial_df.copy()
        X['month'] = X.index.month
        X['year'] = X.index.year
        X['quarter'] = X.index.quarter

        # Ajouter le prix du blé décalé (lag)
        X['Prix_Ble_Lag1'] = y_series.shift(1)

        # Gérer le NaN créé par le shift
        X_processed = X.dropna().copy()
        y_processed = y_series[X_processed.index].copy()

        if len(X_processed) == 0:
            st.error("ERREUR : Aucune donnée disponible après la gestion des valeurs manquantes et du lag. Le fichier est peut-être vide ou corrompu.")
            return None, None, None

        return X_processed, y_processed, X_initial_df.columns.tolist() # Retourne aussi les noms des facteurs initiaux


    X_processed, y_processed, initial_factor_names = engineer_features(X_initial, y)

    if X_processed is not None: # Continuer seulement si l'ingénierie a réussi

        # --- Liste des Facteurs Utilisés ---
        st.subheader("Facteurs Utilisés pour la Modélisation")
        st.write("Les facteurs suivants, ainsi que des caractéristiques temporelles (mois, année, trimestre) et le prix du blé du mois précédent (Lag1), sont utilisés :")
        if initial_factor_names:
            st.write("- " + "\n- ".join(initial_factor_names))
        else:
            st.write("- *Aucun facteur numérique initial trouvé dans les données.*")
        st.markdown("---")

        # --- Analyse de Corrélation (Tableau uniquement) ---
        st.subheader("Analyse de Corrélation")
        df_numeric = df.select_dtypes(include=np.number)
        if not df_numeric.empty:
            try:
                correlation_matrix = df_numeric.corr()
                # Afficher la corrélation avec la cible
                if target_column_name in correlation_matrix.columns:
                    st.write(f"Corrélation des facteurs numériques avec le {target_column_name} :")
                    # Afficher le tableau de corrélation au lieu de la heatmap
                    st.dataframe(correlation_matrix[[target_column_name]].sort_values(by=target_column_name, ascending=False))
                else:
                     st.warning(f"La colonne cible '{target_column_name}' n'est pas numérique, corrélation non calculée avec les facteurs.")

            except Exception as e:
                st.error(f"ERREUR lors du calcul de la corrélation : {e}")
        else:
            st.info("Pas de données numériques pour calculer la corrélation.")
        st.markdown("---")

        # --- Entraînement du Modèle ---
        @st.cache_resource # Cache le modèle entraîné
        def train_model(X_train_processed, y_train_processed):
            """Entraîne le modèle de Régression Linéaire."""
            model = LinearRegression()
            model.fit(X_train_processed, y_train_processed)
            return model

        final_model = train_model(X_processed, y_processed)

        # --- Préparation et Prédiction Future ---

        def predict_future(model, X_processed_df, y_processed_series, num_prediction_months, target_col, initial_factor_cols):
            """Prédit les prix futurs de manière itérative."""
            last_date = X_processed_df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=num_prediction_months, freq='MS')

            X_future_base = pd.DataFrame(index=future_dates)
            X_future_base['month'] = X_future_base.index.month
            X_future_base['year'] = X_future_base.index.year
            X_future_base['quarter'] = X_future_base.index.quarter

            # Utiliser les dernières valeurs connues des facteurs initiaux
            last_known_factors = X_processed_df[initial_factor_cols].iloc[-1]
            for factor in initial_factor_cols:
                 X_future_base[factor] = last_known_factors[factor]

            # Assurer l'ordre des colonnes pour la prédiction (sans le lag initial)
            cols_order_no_lag = X_processed_df.drop(columns=['Prix_Ble_Lag1']).columns
            X_future_base = X_future_base.reindex(columns=cols_order_no_lag, fill_value=0)

            future_predictions = []
            current_lagged_price = y_processed_series.iloc[-1]
            cols_order = X_processed_df.columns

            for i in range(num_prediction_months):
                current_month_features = X_future_base.iloc[i].copy()
                current_month_features['Prix_Ble_Lag1'] = current_lagged_price

                predict_features_dict = current_month_features.to_dict()
                current_month_features_ordered = pd.DataFrame([predict_features_dict], index=[future_dates[i]])
                current_month_features_ordered = current_month_features_ordered.reindex(columns=cols_order, fill_value=0)

                predicted_price = model.predict(current_month_features_ordered)[0]
                future_predictions.append(predicted_price)
                current_lagged_price = predicted_price

            future_results = pd.DataFrame({date_column_name: future_dates, target_col: future_predictions})
            future_results.set_index(date_column_name, inplace=True)

            return future_results

        future_predictions_df = predict_future(
            final_model,
            X_processed,
            y_processed,
            n_prediction_months,
            target_column_name,
            initial_factor_names
        )

        # --- Affichage des Résultats ---

        st.subheader("Prédictions du Prix du Blé")

        # Combiner données historiques et prédictions pour le graphique
        # On va renommer les colonnes pour st.line_chart puisse les distinguer
        historical_data_plot = df[[target_column_name]].rename(columns={target_column_name: 'Historique (Réel)'})
        future_predictions_plot = future_predictions_df[[target_column_name]].rename(columns={target_column_name: 'Prédiction'})

        # Concaténer les deux dataframes
        # Utiliser join='outer' pour gérer les indices de date différents
        combined_data_plot = historical_data_plot.join(future_predictions_plot, how='outer')


        # Visualiser les résultats avec st.line_chart
        st.write("Graphique du Prix du Blé (Historique et Prévision) :")
        # st.line_chart prend directement un DataFrame où l'index est l'axe x
        st.line_chart(combined_data_plot)

        st.markdown("---")

        # Tableau des prédictions futures
        st.write("Tableau des Prédictions Futures :")
        st.dataframe(future_predictions_df)

        st.markdown("---")
        st.write("Méthode : Régression Linéaire avec lag du prix précédent. Les valeurs futures des facteurs initiaux sont basées sur la dernière valeur connue.")
        st.write("Note : Les graphiques sont générés avec les fonctions intégrées de Streamlit.")
