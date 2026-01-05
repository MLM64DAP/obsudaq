import geopandas as gpd

path_gpkg = "Data_input/CARTO/DONNEES_DEP/dep.gpkg"

gdf = gpd.read_file(path_gpkg)

# 3. Reconstruire les départements (dissolve cantons → département)
gdf_dep = gdf.dissolve(
    by="code_insee_du_departement",
    as_index=False
)

# 4. Nettoyage colonnes
gdf_dep = gdf_dep.rename(columns={
    "code_insee_du_departement": "DEP",
    "nom_officiel": "NOM_DEP"
})

# 5. Reprojection WGS84
gdf_dep = gdf_dep.to_crs(4326)

# 6. Simplification géométrique (gros gain de taille)
gdf_dep["geometry"] = gdf_dep.geometry.simplify(
    tolerance=0.01,   # ≈ 1 km
    preserve_topology=True
)

# 7. Export léger pour Streamlit
gdf_dep.to_file("dep_light.geojson", driver="GeoJSON")

print("✅ dep_light.geojson créé")
print("Taille finale (features) :", len(gdf_dep))