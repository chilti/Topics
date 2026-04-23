import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_alluvial_fronts(df_transitions):
    """
    Crea un diagrama de aluvión (Sankey) para mostrar la evolución de los frentes.
    """
    if df_transitions.empty:
        return None
        
    # Mapeo de nodos para Sankey
    # Nodo = (Bin, ClusterID)
    all_nodes = []
    for _, row in df_transitions.iterrows():
        all_nodes.append(f"B{row['from_bin']}_C{row['from_cluster']}")
        all_nodes.append(f"B{row['to_bin']}_C{row['to_cluster']}")
        
    unique_nodes = sorted(list(set(all_nodes)))
    node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
    
    sources = [node_to_idx[f"B{row['from_bin']}_C{row['from_cluster']}"] for _, row in df_transitions.iterrows()]
    targets = [node_to_idx[f"B{row['to_bin']}_C{row['to_cluster']}"] for _, row in df_transitions.iterrows()]
    values = df_transitions['shared_docs'].tolist()
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = unique_nodes,
          color = "blue"
        ),
        link = dict(
          source = sources,
          target = targets,
          value = values
        ))])
        
    fig.update_layout(title_text="Evolución de Frentes de Investigación", font_size=10)
    return fig

def plot_triple_view(df):
    """
    Crea una comparativa side-by-side de los clusters estructurales y semánticos.
    """
    # Para la vista semántica usamos un scatter de las proyecciones UMAP (si existen)
    # Por ahora, si no tenemos las proyecciones en el DF, no podemos graficar.
    # Pero podemos hacer un scatter usando alguna otra métrica o simplemente una tabla.
    
    # Placeholder: Scatter Plot de clusters
    fig = px.scatter(
        df, 
        x="publication_year", 
        y="cluster_semantic", 
        color="cluster_leiden",
        hover_data=["title"],
        title="Consistencia Estructural vs Semántica"
    )
    return fig
