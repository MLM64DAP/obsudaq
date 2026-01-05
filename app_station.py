# =====================================================
# IMPORTS
# =====================================================
import zipfile
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import calendar
from pathlib import Path
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
import matplotlib.lines as mlines

# =====================================================
# PARAMÈTRES
# =====================================================
deps = ["32","40","64","65"]
horizons = ["Horizon30","Horizon50","Horizon100"]
scenarios = ["MIN","Q50","MAX"]
drias_dir = Path("Data_input/DRIAS")
drias_indics = [
    "NORTMm_yr","NORTXm_seas_JJA","NORTX35D_yr",
    "NORTR_yr","NORRR_yr","NORRx1d_yr","NORRRq99_yr",
    "Latitude","Longitude"
]
horizon_to_year = {"Horizon30":2030,"Horizon50":2050,"Horizon100":2080}

# =====================================================
# 1️⃣ Lecture et traitement des données Météo
# =====================================================
def load_meteo(deps, start_year=1980):
    dfs = []

    zip_path = "Data_input/METEO/MENSUELLE/MENSQ_dep.zip"

    # Colonnes de jours de pluie/neige
    nbj_cols = ['NBJRR1','NBJRR5','NBJRR10','NBJRR30','NBJRR50','NBJNEIG']
    # Colonnes de température
    temp_mean_cols = ['TX','TN','TM']
    temp_max_cols = ['TXAB']
    temp_sum_cols = ['NBJTX35','NBJTNS20','NBJTN5']

    with zipfile.ZipFile(zip_path) as z:
        for dep in deps:
            # chercher le bon CSV dans le zip
            matches = [
                name for name in z.namelist()
                if name.endswith(".csv") and f"_{dep}_" in name
            ]

            if not matches:
                print(f"⚠️ Département {dep} absent du ZIP METEO")
                continue

            csv_name = matches[0]

            with z.open(csv_name) as f:
                df = pd.read_csv(f, sep=';')

            df['DEP'] = dep
            df['AAAAMM'] = df['AAAAMM'].astype(str)
            df['ANNEE'] = df['AAAAMM'].str[:4].astype(int)
            df['MOIS'] = df['AAAAMM'].str[4:6].astype(int)
            df = df[df['ANNEE'] >= start_year]

            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("Aucune donnée METEO chargée")

    meteo = pd.concat(dfs, ignore_index=True)

    # =====================================================
    # AGRÉGATIONS ANNUELLES (INCHANGÉ)
    # =====================================================
    annual_parts = {}

    if 'RR' in meteo.columns:
        annual_parts['prec'] = (
            meteo.groupby(['DEP','NUM_POSTE','NOM_USUEL','LAT','LON','ANNEE'])['RR']
            .sum().reset_index()
            .rename(columns={'RR':'cumul_prec_mm'})
        )
    else:
        annual_parts['prec'] = pd.DataFrame()

    nbj_existing = [c for c in nbj_cols if c in meteo.columns]
    if nbj_existing:
        annual_parts['days'] = (
            meteo.groupby(['DEP','NUM_POSTE','NOM_USUEL','LAT','LON','ANNEE'])[nbj_existing]
            .sum().reset_index()
        )
    else:
        annual_parts['days'] = pd.DataFrame()

    temp_mean_existing = [c for c in temp_mean_cols if c in meteo.columns]
    if temp_mean_existing:
        annual_parts['temp_mean'] = (
            meteo.groupby(['DEP','NUM_POSTE','NOM_USUEL','LAT','LON','ANNEE'])[temp_mean_existing]
            .mean().reset_index()
        )
    else:
        annual_parts['temp_mean'] = pd.DataFrame()

    temp_max_existing = [c for c in temp_max_cols if c in meteo.columns]
    if temp_max_existing:
        annual_parts['txab'] = (
            meteo.groupby(['DEP','NUM_POSTE','NOM_USUEL','LAT','LON','ANNEE'])[temp_max_existing]
            .max().reset_index()
        )
    else:
        annual_parts['txab'] = pd.DataFrame()

    temp_sum_existing = [c for c in temp_sum_cols if c in meteo.columns]
    if temp_sum_existing:
        annual_parts['temp_sum'] = (
            meteo.groupby(['DEP','NUM_POSTE','NOM_USUEL','LAT','LON','ANNEE'])[temp_sum_existing]
            .sum().reset_index()
        )
    else:
        annual_parts['temp_sum'] = pd.DataFrame()

    annual = None
    for part in ['prec','days','temp_mean','txab','temp_sum']:
        df_part = annual_parts[part]
        if df_part.empty:
            continue
        annual = df_part if annual is None else annual.merge(
            df_part,
            on=['DEP','NUM_POSTE','NOM_USUEL','LAT','LON','ANNEE'],
            how='outer'
        )

    if 'RRAB' in meteo.columns:
        annual_rrab = (
            meteo.groupby(['DEP','NUM_POSTE','NOM_USUEL','LAT','LON','ANNEE'])['RRAB']
            .max().reset_index()
            .rename(columns={'RRAB':'record_rrab_mm'})
        )
        annual = annual.merge(
            annual_rrab,
            on=['DEP','NUM_POSTE','NOM_USUEL','LAT','LON','ANNEE'],
            how='left'
        )

    annual['jours_par_an'] = annual['ANNEE'].apply(
        lambda y: 366 if calendar.isleap(y) else 365
    )
    for col in nbj_existing:
        annual[f'jours_sans_{col}'] = annual['jours_par_an'] - annual[col]

    if not annual.empty:
        stations_recentes = annual.loc[
            annual['ANNEE'] == annual['ANNEE'].max(), 'NOM_USUEL'
        ].unique()
        annual = annual[annual['NOM_USUEL'].isin(stations_recentes)]

    return annual

# Charger les données
annual = load_meteo(deps)

# =====================================================
# 2️⃣ Lecture des fichiers DRIAS
# =====================================================
def read_drias_file(filepath):
    with open(filepath,"r",encoding="utf-8") as f:
        lines = f.readlines()
    header_line = next((i for i,l in enumerate(lines) if l.strip().startswith("Point;Latitude;Longitude;Niveau")), None)
    if header_line is None:
        raise ValueError(f"Impossible de trouver l'en-tête dans {filepath.name}")
    df = pd.read_csv(filepath, sep=";", skiprows=header_line, comment="#", engine="python")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")].dropna(axis=1, how='all')
    return df

def load_drias(deps,horizons,scenarios,drias_dir,keep_cols=drias_indics):
    dfs = []
    for dep in deps:
        for hor in horizons:
            for scen in scenarios:
                pattern = drias_dir/dep/f"indices{scen}_*_{hor}_{dep}.txt"
                files = list(pattern.parent.glob(pattern.name))
                if not files: 
                    print(f"⚠️ Fichier manquant : DEP {dep}, {hor}, {scen}")
                    continue
                try:
                    df = read_drias_file(files[0])
                    keep = [c for c in keep_cols if c in df.columns]
                    if not keep: continue
                    df = df[keep]
                    df['DEP'] = dep
                    df['HORIZON'] = hor
                    df['SCENARIO'] = scen
                    dfs.append(df)
                except Exception as e:
                    print(f"❌ Erreur lecture {files[0].name}: {e}")
    if not dfs: raise FileNotFoundError("Aucun fichier DRIAS valide trouvé")
    drias_all = pd.concat(dfs, ignore_index=True)
    drias_all = drias_all.loc[:, ~drias_all.columns.str.contains('^Unnamed')].dropna(axis=1, how='all')
    return drias_all

drias_all = load_drias(deps,horizons,scenarios,drias_dir)

# =====================================================
# 3️⃣ Géopandas : Stations + DRIAS + Départements
# =====================================================
dep_path = Path("Data_input") / "CARTO" / "DONNEES_DEP" / "dep.gpkg"

gdf_dep = gpd.read_file(dep_path)

fr_cible = gdf_dep[gdf_dep['code_insee_du_departement'].astype(str).isin(deps)]

gdf_drias = gpd.GeoDataFrame(
    drias_all,
    geometry=gpd.points_from_xy(drias_all["Longitude"],drias_all["Latitude"]),
    crs="EPSG:4326"
).to_crs(fr_cible.crs)

gdf_stations = gpd.GeoDataFrame(
    annual,
    geometry=gpd.points_from_xy(annual["LON"],annual["LAT"]),
    crs="EPSG:4326"
).to_crs(fr_cible.crs)

# =====================================================
# 4️⃣ Association stations -> points DRIAS (nearest)
# =====================================================
nearest = gpd.sjoin_nearest(gdf_stations,gdf_drias,how="left",distance_col="dist_m")

# Créer explicitement les colonnes manquantes
nearest["DRIAS_LAT"] = nearest.geometry.y
nearest["DRIAS_LON"] = nearest.geometry.x
nearest["DRIAS_DEP"] = nearest.get("DEP_right", np.nan)
nearest["DEP"] = nearest.get("DEP_left", nearest.get("DEP"))

# Renommer indicateurs DRIAS
drias_indic_cols = ["NORRR_yr","NORRx1d_yr","NORTMm_yr"]
nearest = nearest.rename(columns={c:f"DRIAS_{c}" for c in drias_indic_cols})

# =====================================================
# 5️⃣ Préparer obs et projections
# =====================================================
obs_cols = ['NOM_USUEL','ANNEE','cumul_prec_mm','record_rrab_mm','NBJRR1','NBJRR5','NBJRR10','NBJRR30','NBJRR50']
obs = nearest[obs_cols].rename(columns={'cumul_prec_mm':'OBS_cumul_prec_mm','record_rrab_mm':'OBS_rrmax24h_mm'})
obs = obs.drop_duplicates(subset=['NOM_USUEL','ANNEE']).reset_index(drop=True)

drias_cols = ['NOM_USUEL','HORIZON','SCENARIO','DRIAS_DEP','DRIAS_LAT','DRIAS_LON','DRIAS_NORRR_yr','DRIAS_NORRx1d_yr','DRIAS_NORTMm_yr','dist_m']
drias_proj = nearest[drias_cols].rename(columns={'DRIAS_NORRR_yr':'DRIAS_cumul_prec_mm','DRIAS_NORRx1d_yr':'DRIAS_rrmax24h_mm','DRIAS_NORTMm_yr':'DRIAS_temp_moy_yr'})
drias_proj['DRIA_ANNEE'] = drias_proj['HORIZON'].map(horizon_to_year)

# =====================================================
# 6️⃣ FONCTION DE PLOT INTERACTIF (Streamlit ou notebook)
# =====================================================
def plot_station_precip(station_name, obs=obs, drias_proj=drias_proj):
    df_obs = obs[obs['NOM_USUEL']==station_name]
    df_proj = drias_proj[drias_proj['NOM_USUEL']==station_name]
    if df_obs.empty: return None

    # Cumul annuel
    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(df_obs['ANNEE'], df_obs['OBS_cumul_prec_mm'], label='Obs cumul précip (mm)', color='tab:blue', marker='o')
    for scenario,color in zip(['MIN','Q50','MAX'],['tab:green','tab:orange','tab:red']):
        grp = df_proj[df_proj['SCENARIO']==scenario]
        if not grp.empty:
            grp_mean = grp.groupby('DRIA_ANNEE')['DRIAS_cumul_prec_mm'].mean().reset_index()
            ax1.scatter(grp_mean['DRIA_ANNEE'], grp_mean['DRIAS_cumul_prec_mm'], label=f'DRIAS cumul {scenario}', color=color, marker='s')
    ax1.set_xlabel('Année'); ax1.set_ylabel('Cumul annuel (mm)'); ax1.grid(axis='y', alpha=0.3); ax1.legend(); ax1.set_title(station_name)
    
    # Max 24h
    fig2, ax2 = plt.subplots(figsize=(12,5))
    ax2.plot(df_obs['ANNEE'], df_obs['OBS_rrmax24h_mm'], label='Obs max 24h', color='tab:red', marker='^')
    for scenario,color in zip(['MIN','Q50','MAX'],['tab:green','tab:orange','tab:red']):
        grp = df_proj[df_proj['SCENARIO']==scenario]
        if not grp.empty:
            grp_mean = grp.groupby('DRIA_ANNEE')['DRIAS_rrmax24h_mm'].mean().reset_index()
            ax2.scatter(grp_mean['DRIA_ANNEE'], grp_mean['DRIAS_rrmax24h_mm'], label=f'DRIAS max 24h {scenario}', color=color, marker='v')
    ax2.set_xlabel('Année'); ax2.set_ylabel('Max 24h (mm)'); ax2.grid(axis='y', alpha=0.3); ax2.legend(); ax2.set_title(station_name)
    
    return fig1, fig2

# =====================================================
# 7️⃣ STREAMLIT
# =====================================================
st.set_page_config(layout="wide", page_title="Météo et DRIAS")
page = st.sidebar.radio("Navigation", ["Description","Historique Précipitations","Précipitations DRIAS", "Températures"])

if page=="Description":
    st.title("Description")
    st.markdown("""
    # Observations et Projections Climatiques en France
    **Projet : Analyse et visualisation des données climatiques par station**  
    **Auteur : Marc Le Moing**  
    **Date : Décembre 2025**

    ---

    ## Contexte et objectif
    Ce projet a pour objectif de **collecter, traiter et visualiser les données climatiques observées et projetées** sur le territoire français.  
    Il permet de suivre l’évolution des températures, des précipitations et des événements extrêmes (jours >35°C, nuits tropicales, jours froids, maximums de pluie) au niveau local (stations météorologiques) et régional (départements).  
    
    ---

    ## Sources de données
    - **Observations historiques (1980–2023)** :  
        - Températures maximales (TX), minimales (TN), moyennes (TM)  
        - Précipitations cumulées annuelles et maximum 24h  
        - Indicateurs extrêmes : jours >35°C, nuits tropicales, jours froids  
    - **Projections climatiques DRIAS** :  
        - Températures et précipitations pour 2030, 2050 et 2100  
        - Scénarios MIN, Q50, MAX pour estimer incertitudes  
    - **Métadonnées des stations** : 
        - NOM_USUEL, DEP pour géolocaliser chaque observation


    ---

    ## Démarche méthodologique
    1. **Collecte et nettoyage des données** : agrégation annuelle par station et département, calcul des indicateurs extrêmes, harmonisation des projections DRIAS.  
    2. **Analyse exploratoire** : tendances temporelles, moyennes mobiles 5 ans, régressions linéaires pour prolonger les tendances.  
    3. **Visualisation interactive** : interface Streamlit pour sélectionner un département ou une station, visualisation combinant observations et projections DRIAS avec incertitudes.  
    4. **Indicateurs clés** :  
        - Températures : TX, TM, TN  
        - Jours >35°C, nuits tropicales (>20°C), jours froids (≤ -5°C)  
        - Précipitations : cumul annuel, maximum 24h
    ---

    ## Études réalisées
    - Analyse des tendances climatiques par département et par station.  
    - Comparaison historique vs projections DRIAS pour 2030, 2050 et 2100.  
    - Étude des indicateurs extrêmes pour comprendre l’exposition locale.  
    - Visualisation multi-station et multi-département pour comparaison régionale.

    ---

    ## Limites
    - Séries historiques parfois incomplètes.  
    - Projections DRIAS à l’échelle départementale, approximation locale.  
    - Moyennes mobiles lissant certaines variations.  
    - Rareté des événements extrêmes pouvant générer des biais.

    ---

    ## Perspectives
    - Intégration des données pluviométriques et vent pour enrichir l’analyse.  
    - Modèles à maille fine pour meilleure précision spatiale des projections.  
    - Alertes automatiques pour événements extrêmes par station.  
    - Tableau de bord interactif multi-stations et départements.  
    - Extension à l’ensemble du territoire européen pour comparaison transnationale.

    ---

    ## Conclusion
    Ce projet fournit un outil robuste pour **observer et projeter les températures et indicateurs extrêmes** en France.  
    Les analyses et visualisations interactives permettent de mieux comprendre le **changement climatique et ses impacts locaux et régionaux**.
    """)


elif page=="Historique Précipitations":
    st.title("Historique des précipitations et jours de neige")
    
    # Sélection département
    dep_sel = st.selectbox("Département", ["Tous"] + sorted(annual.DEP.unique()), key="hist_dep")
    
    # Filtrer stations selon département
    if dep_sel != "Tous":
        stations_list = sorted(annual[annual.DEP == dep_sel].NOM_USUEL.unique())
    else:
        stations_list = sorted(annual.NOM_USUEL.unique())
    
    # Sélection station : on prend la première de la liste par défaut
    station_sel = st.selectbox("Station", stations_list, index=0, key="hist_poste")
    
    # Filtrer le DataFrame selon sélection
    df_plot = annual.copy()
    if dep_sel != "Tous":
        df_plot = df_plot[df_plot.DEP == dep_sel]
    if station_sel:
        df_plot = df_plot[df_plot.NOM_USUEL == station_sel]
    
    if df_plot.empty:
        st.warning("Pas de données pour cette sélection")
    else:
        # Graphique cumul précip + nb jours pluie
        fig, ax1 = plt.subplots(figsize=(12,6))
        ax1.bar(df_plot.ANNEE, df_plot.cumul_prec_mm, alpha=0.6, label="Cumul annuel (mm)")
        ax1.set_ylabel("Cumul annuel (mm)")
        ax1.set_xlabel("Année")
        ax1.set_title(f"{station_sel} - Cumul précipitations et nombre de jours de pluie")
        
        if "NBJRR1" in df_plot.columns:
            ax2 = ax1.twinx()
            ax2.plot(df_plot.ANNEE, df_plot.NBJRR1, color="tab:orange", marker="o", label="Nb jours pluie")
            h1,l1 = ax1.get_legend_handles_labels()
            h2,l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1+h2, l1+l2, loc="upper left")
        else:
            ax1.legend(loc="upper left")
        st.pyplot(fig)

        # Graphique nb jours de neige
        if "NBJNEIG" in df_plot.columns:
            fig2, ax = plt.subplots(figsize=(12,5))
            ax.plot(df_plot.ANNEE, df_plot.NBJNEIG, color="tab:blue", marker="o")
            ax.set_ylabel("Nb jours de neige")
            ax.set_xlabel("Année")
            ax.set_title(f"{station_sel} - Nombre de jours de neige")
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig2)


elif page == "Précipitations DRIAS":
    st.title("Précipitations DRIAS")

    # Sélection département à partir des projections
    dep_list = sorted(drias_proj.DRIAS_DEP.unique())
    dep_sel = st.selectbox("Département", ["Tous"] + dep_list, key="drias_dep")

    # Filtrer stations selon département (obs pour afficher les stations)
    if dep_sel != "Tous":
        stations_list = sorted(obs[obs.NOM_USUEL.isin(
            drias_proj[drias_proj.DRIAS_DEP == dep_sel]["NOM_USUEL"]
        )]["NOM_USUEL"].unique())
    else:
        stations_list = sorted(obs["NOM_USUEL"].unique())

    # Sélection station
    station_sel = st.selectbox("Station", stations_list, key="drias_station")

    # Filtrer observations et projections
    df_obs = obs[obs.NOM_USUEL == station_sel]
    df_proj = drias_proj[drias_proj.NOM_USUEL == station_sel]

    if dep_sel != "Tous":
        df_proj = df_proj[df_proj.DRIAS_DEP == dep_sel]  # filtre projections

    if df_obs.empty:
        st.warning("Pas de données pour cette station")
    else:
        # Cumul annuel
        fig1, ax1 = plt.subplots(figsize=(12,5))
        ax1.plot(df_obs.ANNEE, df_obs.OBS_cumul_prec_mm, label="Obs cumul précip (mm)", color="tab:blue", marker="o", linestyle="-")
        for scenario in ["MIN","Q50","MAX"]:
            grp = df_proj[df_proj.SCENARIO == scenario]
            if not grp.empty:
                grp_mean = grp.groupby("DRIA_ANNEE")["DRIAS_cumul_prec_mm"].mean().reset_index()
                ax1.scatter(grp_mean.DRIA_ANNEE, grp_mean.DRIAS_cumul_prec_mm,
                            label=f"DRIAS cumul {scenario}", marker="s",
                            color={"MIN":"tab:green","Q50":"tab:orange","MAX":"tab:red"}[scenario])
        ax1.set_xlabel("Année")
        ax1.set_ylabel("Cumul annuel (mm)")
        ax1.set_title(f"{station_sel} - Cumul annuel précipitations")
        ax1.grid(axis="y", alpha=0.3)
        ax1.legend()
        st.pyplot(fig1)

        # Max 24h
        fig2, ax2 = plt.subplots(figsize=(12,5))
        ax2.plot(df_obs.ANNEE, df_obs.OBS_rrmax24h_mm, label="Obs max 24h", color="tab:red", marker="^", linestyle="-")
        for scenario in ["MIN","Q50","MAX"]:
            grp = df_proj[df_proj.SCENARIO == scenario]
            if not grp.empty:
                grp_mean = grp.groupby("DRIA_ANNEE")["DRIAS_rrmax24h_mm"].mean().reset_index()
                ax2.scatter(grp_mean.DRIA_ANNEE, grp_mean.DRIAS_rrmax24h_mm,
                            label=f"DRIAS max 24h {scenario}", marker="v",
                            color={"MIN":"tab:green","Q50":"tab:orange","MAX":"tab:red"}[scenario])
        ax2.set_xlabel("Année")
        ax2.set_ylabel("Max 24h (mm)")
        ax2.set_title(f"{station_sel} - Max 24h précipitations")
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)


elif page == "Températures":
    st.title("Évolution des températures et projections DRIAS par station")

    # ===============================
    # Sélection département et station
    # ===============================
    dep_sel = st.selectbox("Département", ["Tous"] + sorted(annual.DEP.unique()), key="temp_dep")

    # Filtrer les stations selon le département
    if dep_sel != "Tous":
        stations_list = sorted(annual[annual.DEP == dep_sel].NOM_USUEL.unique())
    else:
        stations_list = sorted(annual.NOM_USUEL.unique())

    # Sélection station
    station_sel = st.selectbox("Station", stations_list, key="temp_station")

    # Historique pour la station
    df_hist = annual[annual["NOM_USUEL"] == station_sel].sort_values("ANNEE").copy()

    if df_hist.empty:
        st.warning("Pas de données pour cette station")
    else:
        # Moyenne mobile 5 ans sur TM
        df_hist["ma5_TM"] = df_hist["TM"].rolling(window=5, min_periods=1, center=True).mean()

        # Régression linéaire TM
        df_fit = df_hist[['ANNEE','ma5_TM']].dropna()
        X = df_fit['ANNEE'].values.reshape(-1,1)
        y = df_fit['ma5_TM'].values
        if len(y) > 1:
            model = LinearRegression().fit(X, y)
            x_future = np.arange(df_fit['ANNEE'].min(), 2110).reshape(-1,1)
            y_future = model.predict(x_future)
        else:
            x_future, y_future = np.array([]), np.array([])

        # ===============================
        # Graphique Températures annuelles
        # ===============================
        fig, ax = plt.subplots(figsize=(12,6))

        for col, color, label in zip(['TX','TM','TN'], ['tomato','orange','royalblue'], ['TX (max)','TM (moy)','TN (min)']):
            if col in df_hist.columns:
                ax.plot(df_hist["ANNEE"], df_hist[col], color=color, alpha=0.6, label=label)

        # Moyenne mobile TM
        ax.plot(df_hist["ANNEE"], df_hist["ma5_TM"], color="black", linewidth=2, label="TM moyenne mobile 5 ans")

        # Tendance prolongée
        if len(x_future) > 0:
            ax.plot(x_future, y_future, linestyle="--", color="red", alpha=0.8, label="Tendance prolongée")

        # Projections DRIAS pour TM
        proj = drias_proj[drias_proj["NOM_USUEL"] == station_sel]
        if not proj.empty:
            proj_summary = proj.pivot_table(index="HORIZON", columns="SCENARIO", values="DRIAS_temp_moy_yr").reset_index()
            colors = {"Horizon30": "orange", "Horizon50": "red", "Horizon100": "darkred"}
            mapping_horizon_to_year = {"Horizon30":2030, "Horizon50":2050, "Horizon100":2100}

            for _, row in proj_summary.iterrows():
                horizon = row["HORIZON"]
                year = mapping_horizon_to_year.get(horizon)
                color = colors.get(horizon, "gray")
                if year is not None:
                    if not pd.isna(row.get("MIN")) and not pd.isna(row.get("MAX")):
                        ax.vlines(year, row["MIN"], row["MAX"], color=color, linewidth=3, alpha=0.5)
                    if not pd.isna(row.get("Q50")):
                        ax.scatter(year, row["Q50"], color=color, edgecolor="k", s=140, zorder=5,
                                   label=f"{horizon} ({year}) : {row['Q50']:.1f} °C")

        # Repères verticaux horizons
        for yv in [2030, 2050, 2100]:
            ax.axvline(yv, color="gray", linestyle=":", alpha=0.3)

        # Légende barre incertitude
        bar_legend = mlines.Line2D([], [], color='k', linewidth=3, alpha=0.5, label='Barre verticale = incertitude (MIN–MAX)')
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(1, bar_legend)
        ax.legend(handles=handles, loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

        ax.set_title(f"{station_sel} — Températures annuelles et projections DRIAS")
        ax.set_xlabel("Année")
        ax.set_ylabel("Température (°C)")
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)

        # =====================================================
        # INDICATEURS DE TEMPÉRATURES EXTRÊMES (Jours TX>35, Nuits tropicales, Jours froids)
        # =====================================================
        st.subheader("Indicateurs de températures extrêmes")

        indicateurs = [
            ("NBJTX35", "NORTX35D_yr", "Jours > 35°C", "tomato"),
            ("NBJTNS20", "NORTR_yr", "Nuits tropicales (>20°C)", "orange"),
            ("NBJTN5", "NORTN5D_yr", "Jours froids (≤ -5°C)", "royalblue")
        ]

        df_ext = annual[annual["NOM_USUEL"] == station_sel].sort_values("ANNEE").copy()
        proj_ext = drias_all[drias_all["DEP"].isin(df_ext["DEP"].unique())]

        fig_ext, axes = plt.subplots(len(indicateurs), 1, figsize=(12, 4 * len(indicateurs)), sharex=True)
        if len(indicateurs) == 1:
            axes = [axes]

        for ax, (col_hist, col_drias, titre, color) in zip(axes, indicateurs):
            # Moyenne mobile 5 ans
            df_ext[f"ma5_{col_hist}"] = df_ext[col_hist].rolling(window=5, min_periods=1, center=True).mean()
            
            # Régression linéaire
            df_fit = df_ext[['ANNEE', f"ma5_{col_hist}"]].dropna()
            if len(df_fit) > 1:
                X = df_fit['ANNEE'].values.reshape(-1, 1)
                y = df_fit[f"ma5_{col_hist}"].values
                model = LinearRegression().fit(X, y)
                x_future = np.arange(df_fit['ANNEE'].min(), 2101).reshape(-1, 1)
                y_future = model.predict(x_future)
                ax.plot(x_future, y_future, "--", color=color, alpha=0.7, label="Tendance prolongée")

            # Historique
            ax.plot(df_ext["ANNEE"], df_ext[f"ma5_{col_hist}"], color=color, marker="o", label="Moyenne mobile 5 ans")

            # Projections DRIAS
            if col_drias in proj_ext.columns:
                proj_summary = proj_ext.pivot_table(index="HORIZON", columns="SCENARIO", values=col_drias).reset_index()
                colors_h = {"Horizon30": "orange", "Horizon50": "red", "Horizon100": "darkred"}
                years_h = {"Horizon30":2030, "Horizon50":2050, "Horizon100":2100}

                for _, row in proj_summary.iterrows():
                    h = row["HORIZON"]
                    year = years_h.get(h)
                    c = colors_h.get(h, "gray")
                    if year is None: 
                        continue
                    if not pd.isna(row.get("MIN")) and not pd.isna(row.get("MAX")):
                        ax.vlines(year, row["MIN"], row["MAX"], color=c, linewidth=3, alpha=0.5)
                    if not pd.isna(row.get("Q50")):
                        ax.scatter(year, row["Q50"], color=c, edgecolor="k", s=120, label=f"{h} ({year}) : {row['Q50']:.1f}")

            ax.set_title(titre)
            ax.set_ylabel("Nombre de jours / an")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        axes[-1].set_xlabel("Année")
        plt.suptitle(f"{station_sel} — Indicateurs extrêmes (obs + DRIAS)", fontsize=14, y=0.98)
        plt.tight_layout(rect=[0,0,1,0.96])
        st.pyplot(fig_ext)

        # =====================================================
        # ÉVOLUTION DES TX POUR LA STATION SÉLECTIONNÉE
        # =====================================================
        st.subheader(f"Températures maximales annuelles (TX) — Station {station_sel}")

        # Filtrer les données pour la station choisie
        df_tx_station = annual[annual['NOM_USUEL'] == station_sel].sort_values('ANNEE')

        fig_tx_station, ax_tx_station = plt.subplots(figsize=(14,6))
        if 'TXAB' in df_tx_station.columns:
            ax_tx_station.plot(df_tx_station['ANNEE'], df_tx_station['TXAB'], marker='o', alpha=0.7, label=station_sel)

        ax_tx_station.set_title(f"Évolution des températures maximales annuelles (TX) — {station_sel}")
        ax_tx_station.set_xlabel("Année")
        ax_tx_station.set_ylabel("Température max (°C)")
        ax_tx_station.grid(True, alpha=0.3)
        ax_tx_station.legend(fontsize=8)
        st.pyplot(fig_tx_station)