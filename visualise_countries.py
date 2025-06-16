import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from shapely.geometry import MultiPolygon
import io


# Load shapefile
shapefile_path = 'Europe/ne_110m_admin_0_countries.shp'
gdf = gpd.read_file(shapefile_path)

# Filter to Europe
if 'CONTINENT' in gdf.columns:
    gdf = gdf[gdf['CONTINENT'] == 'Europe']
else:
    europe_country_list = ['Belgium', 'Romania', 'France', 'Germany', 'Netherlands']  # Extend as needed
    gdf = gdf[gdf['NAME'].isin(europe_country_list)]

# Drop Russia
gdf = gdf[gdf['NAME'] != 'Russia']

# Remove southwest part of France
france_idx = gdf[gdf['NAME'] == 'France'].index[0]
france_geom = gdf.loc[france_idx, 'geometry']
if isinstance(france_geom, MultiPolygon):
    polygons = list(france_geom.geoms)
    centroids = [poly.centroid for poly in polygons]
    sw_idx = sorted(range(len(centroids)), key=lambda i: (centroids[i].y, centroids[i].x))[0]
    new_polygons = [p for i, p in enumerate(polygons) if i != sw_idx]
    gdf.at[france_idx, 'geometry'] = new_polygons[0] if len(new_polygons) == 1 else MultiPolygon(new_polygons)

# Load country attributes
csv_path = 'Flexbalance revenue estimates per country and asset - May 2025.xlsx - Country conclusions.csv'
df = pd.read_csv(csv_path)
gdf = gdf.merge(df, how='left', left_on='NAME', right_on='Country')

# Streamlit UI
st.title("Flexbalance Country Potential")
df_columns = [col for col in df.columns if col != 'Country']
column = st.selectbox("Select attribute to visualize", df_columns)

# Hover data: show all attributes
hover_columns = [
    "market depth in MW (to 40k per mw)",
    "market depth in EUR (to 40k per mw)",
    "Renewable (down) revenue (EUR/MW flex/YR)",
    "C&I (up) revenue (EUR/MW flex/YR)",
    "C&I (down) revenue (EUR/MW flex/YR)"
]
hover_data = {col: True for col in hover_columns if col in gdf.columns and col != column}

if pd.api.types.is_numeric_dtype(gdf[column]):
    st.sidebar.subheader("Color Scale Settings")

    col_min = float(gdf[column].min(skipna=True))
    col_max = float(gdf[column].max(skipna=True))
    selected_min, selected_max = st.sidebar.slider(
        f"Value range for '{column}'",
        min_value=col_min - 0.5 * (col_max - col_min),
        max_value=col_max + 0.5 * (col_max - col_min),
        value=(col_min, col_max)
    )

    flip_colors = st.sidebar.checkbox("Flip green and red (reverse scale)", value=False)
    color_scale = "RdYlGn_r" if flip_colors else "RdYlGn"

    # Split GeoDataFrame
    gdf_valid = gdf[gdf[column].notna()]
    gdf_missing = gdf[gdf[column].isna()]

    fig = go.Figure()

    # Trace for valid data with color scale
    fig.add_trace(go.Choropleth(
        geojson=gdf_valid.geometry.__geo_interface__,
        locations=gdf_valid.index,
        z=gdf_valid[column],
        colorscale=color_scale,
        zmin=selected_min,
        zmax=selected_max,
        colorbar_title=column,
        marker_line_color="white",
        hovertext=gdf_valid['NAME'],
        hoverinfo="text+z",
        name="Data"
    ))

    # Trace for missing data in grey
    fig.add_trace(go.Choropleth(
        geojson=gdf_missing.geometry.__geo_interface__,
        locations=gdf_missing.index,
        z=[0] * len(gdf_missing),  # Dummy values
        colorscale=[[0, "#d3d3d3"], [1, "#d3d3d3"]],
        showscale=False,
        marker_line_color="white",
        hovertext=gdf_missing['NAME'],
        hoverinfo="text",
        name="No Data"
    ))

else:
    gdf[column] = gdf[column].fillna("Missing")
    unique_vals = gdf[column].unique()

    palette = px.colors.qualitative.Set2
    color_map = {val: palette[i % len(palette)] for i, val in enumerate(unique_vals) if val != "Missing"}
    color_map["Missing"] = "#d3d3d3"  # Light grey

    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color=column,
        color_discrete_map=color_map,
        hover_name=gdf['NAME'],
        hover_data=hover_data,
        title=f"Choropleth Map of {column}"
    )

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(
    title={
        'text': f"<b>Map of {column}</b>",  # Bold title
        'x': 0.5,  # Center title
        'xanchor': 'center',
        'font': {
            'size': 16,   # Larger font
            'family': 'Arial',
            'color': 'black'
        }
    },
    margin={"r": 0, "t": 40, "l": 0, "b": 0}
)

# Export to PNG
img_bytes = fig.to_image(
    format="png",
    width=2000,
    height=1200,
    scale=2,
    engine="kaleido"
)
buf = io.BytesIO(img_bytes)

st.download_button(
    label="📥 Download high-res map as PNG",
    data=buf,
    file_name=f"{column}_choropleth_map.png",
    mime="image/png"
)

st.plotly_chart(fig)
