import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from Net.Visualizer import NetworkEngine
import os
import sys
import json
import time
import random
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de rutas
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Importar lógica del nuevo pipeline "flat"
from pipeline_topic import (
    get_hierarchy_flat as get_hierarchy, 
    compute_subfield_data_flat as compute_subfield_data, 
    load_subfield_data, 
    load_collaboration_data,
    load_institutional_data,
    load_types_data,
    get_type_distribution,
    load_inst_types_data,
    get_inst_type_distribution,
    get_entity_metrics,
    get_summary_tables
)

# Nuevo: Pipeline de Frentes de Investigación (lanzado como subproceso, no bloquea)
import subprocess

# Configuración de página
st.set_page_config(
    page_title="Dashboard de Tópicos (Flat-Table)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PREMIUM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top right, #fdfdfd, #f4f7f6);
    }

    /* Tarjetas de Métricas Premium */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        flex: 1;
        text-align: left;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }

    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e293b;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

def premium_metric(label, value, delta=None):
    delta_html = ""
    if delta:
        color_class = "delta-positive" if str(delta).startswith("+") else "delta-negative"
        delta_html = f'<div class="metric-delta {color_class}" style="font-size: 0.8rem; font-weight: 500; margin-top: 4px;">{delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# --- CONFIGURATION & MAPPINGS ---
BASE_PATH = Path(__file__).parent
DATA_DIR = BASE_PATH / 'data'
CACHE_TEMAS_DIR = DATA_DIR / 'cache_temas'
CACHE_SUFFIX = "_flat"

INST_METRICS = {
    "Artículos": "doc_count",
    "Impacto (FWCI)": "fwci",
    "% Top 10%": "pct_top_10",
    "% Top 1%": "pct_top_1",
    "Percentil": "percentile"
}

# --- SIDEBAR: JERARQUÍA ---
st.sidebar.title("🧬 Análisis de Tópicos")
st.sidebar.info("Utilizando motor optimizado `works_flat`.")
st.sidebar.markdown("---")

# Hierarchy retrieval
from regions import GLOBAL_REGIONS

df_hier = get_hierarchy()

if df_hier is not None:
    domains = sorted(df_hier['domain'].dropna().unique())
    selected_domain = st.sidebar.selectbox("1. Dominio", domains, index=domains.index('Health Sciences') if 'Health Sciences' in domains else 0)
    
    fields = sorted(df_hier[df_hier['domain'] == selected_domain]['field'].dropna().unique())
    selected_field = st.sidebar.selectbox("2. Campo", fields, index=fields.index('Medicine') if 'Medicine' in fields else 0)
    
    subfields = sorted(df_hier[df_hier['field'] == selected_field]['subfield'].dropna().unique())
    default_sub = "Pulmonary and Respiratory Medicine"
    selected_subfield = st.sidebar.selectbox("3. Subcampo", subfields, index=subfields.index(default_sub) if default_sub in subfields else 0)
else:
    st.sidebar.error("No se pudo cargar la jerarquía de temas desde `works_flat`.")
    st.stop()

st.sidebar.markdown("---")

# Period Selection
period_mode = st.sidebar.radio("Periodo de Análisis", ["Últimos 5 años (2021-2025)", "Periodo Completo"], index=0)

st.sidebar.markdown("---")
show_all_topics = st.sidebar.checkbox("Mostrar todos los tópicos", value=False)

if st.sidebar.button("🔄 Forzar Recálculo (Flat)", help="Borra el caché local y vuelve a consultar ClickHouse"):
    st.session_state.calculating = True
    st.session_state.has_cache = False
    st.rerun()

# --- MAIN CONTENT ---
st.title(f"Tema (Flat): {selected_subfield}")
st.markdown(f"**Dominio:** {selected_domain} | **Campo:** {selected_field}")
st.markdown("---")

# Cache check
if 'selected_subfield' not in st.session_state or st.session_state.selected_subfield != selected_subfield:
    st.session_state.selected_subfield = selected_subfield
    cache_path = CACHE_TEMAS_DIR / f"{selected_subfield.replace(' ', '_').lower()}{CACHE_SUFFIX}.parquet"
    st.session_state.has_cache = cache_path.exists()

if not st.session_state.has_cache:
    st.warning(f"⚠️ Los datos optimizados para '{selected_subfield}' no están calculados.")
    if 'calculating' not in st.session_state:
        st.session_state.calculating = False
    
    if st.button("🚀 Lanzar Cálculo en ClickHouse (Flat)", disabled=st.session_state.calculating):
        st.session_state.calculating = True
        st.rerun()

if not st.session_state.has_cache and st.session_state.get('calculating'):
    with st.spinner("Calculando métricas globales desde works_flat..."):
        success = compute_subfield_data(selected_subfield)
        st.session_state.calculating = False
        if success:
            st.success("¡Cálculo finalizado!")
            st.session_state.has_cache = True
            st.rerun()
        else:
            st.error("No se encontraron datos en works_flat para este subcampo.")
            st.session_state.calculating = False
    st.stop()

# Load Data with suffix
df_data = load_subfield_data(selected_subfield, suffix=CACHE_SUFFIX)
df_collab = load_collaboration_data(selected_subfield, suffix=CACHE_SUFFIX)
df_inst = load_institutional_data(selected_subfield, suffix=CACHE_SUFFIX)
df_types = load_types_data(selected_subfield, suffix=CACHE_SUFFIX)
df_inst_types = load_inst_types_data(selected_subfield, suffix=CACHE_SUFFIX)

# Journals (cache independiente con sufijo)
sub_clean = selected_subfield.strip().replace(' ', '_').lower()
cache_jr = CACHE_TEMAS_DIR / f"{sub_clean}_journals_flat.parquet"
df_journals_top = pd.read_parquet(cache_jr) if cache_jr.exists() else pd.DataFrame(columns=['Revista', 'URL', 'Artículos'])

# --- DEBUG INFO (Solo para diagnóstico) ---
with st.sidebar.expander("🛠️ Debug Cache Status"):
    st.write(f"Suffix: {CACHE_SUFFIX}")
    files = {
        "Main": CACHE_TEMAS_DIR / f"{sub_clean}{CACHE_SUFFIX}.parquet",
        "Inst": CACHE_TEMAS_DIR / f"{sub_clean}_inst{CACHE_SUFFIX}.parquet",
        "Types": CACHE_TEMAS_DIR / f"{sub_clean}_types{CACHE_SUFFIX}.parquet",
        "Jrnl": cache_jr
    }
    for k, v in files.items():
        exists = "✅" if v.exists() else "❌"
        st.write(f"{exists} {k}: {v.name}")

if df_data is None:
    st.error("Error al cargar los datos de caché.")
    st.stop()

# --- DATA AGGREGATION LOGIC ---

def download_csv_button(df, name, use_sidebar=False):
    if df is not None and not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        target = st.sidebar if use_sidebar else st
        target.download_button(
            label=f"📥 Descargar {name}",
            data=csv,
            file_name=f"{name.replace(' ', '_').lower()}_flat.csv",
            mime='text/csv',
            key=f"btn_dl_{name.replace(' ', '_').lower()}_{random.randint(0,99999)}"
        )

def render_entity_kpis(entity_name, df_all, period_label):
    data = get_entity_metrics(df_all, entity_name, period_label)
    if not data or not data['metrics']:
        st.warning(f"No hay suficientes datos para {entity_name}")
        return None
    
    m = data['metrics']
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        premium_metric("Documentos", f"{m['docs']:,}")
        premium_metric("Impacto (FWCI)", f"{m['fwci']:.2f}")
    with c2:
        premium_metric("% Top 10%", f"{m['top_10']:.1f}%")
        premium_metric("% Top 1%", f"{m['top_1']:.2f}%")
    with c3:
        premium_metric("Percentil (Norm)", f"{m['percentile']:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)
    return data

def render_entity_charts_synced(entity_name, data, tab_index):
    if not data or 'trends' not in data or data['trends'].empty:
        st.info(f"Sin tendencias para {entity_name}")
        return

    # Mapeo de índices a métricas
    metrics_map = [
        ('doc_count', 'Producción', '#3b82f6', 'Documentos', True),
        ('fwci', 'Impacto (FWCI)', '#ef4444', 'FWCI', False),
        ('pct_top_10', '% Top 10%', '#8b5cf6', '% Top 10%', False),
        ('pct_top_1', '% Top 1%', '#ec4899', '% Top 1%', False),
        ('percentile', 'Percentil (Normalizado)', '#f59e0b', 'Percentil', False)
    ]
    
    col_name, title_suffix, color, y_label, has_fill = metrics_map[tab_index]
    trends_df = data['trends'][data['trends']['year'] <= 2025].copy()
    
    fig = px.line(trends_df, x='year', y=col_name, 
                  title=f"Evolución {title_suffix}: {entity_name}",
                  labels={col_name: y_label, 'year': 'Año'},
                  markers=True, template="plotly_white")
    
    fig.update_traces(line_color=color)
    if col_name == 'fwci':
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Media Mundial")
        
    try:
        fig.update_xaxes(type='linear', tickformat='d')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error renderizando gráfica: {e}")

def render_topical_evolution(entity_name, data, tab_index, show_all=False):
    """Renderiza la evolución temporal desglosada por tópicos."""
    if not data or 'topical_trends' not in data or data['topical_trends'].empty:
        return

    metrics_map = [
        ('doc_count', 'Producción por Tópico', 'Documentos', True),
        ('fwci', 'FWCI por Tópico', 'FWCI', False),
        ('pct_top_10', '% Top 10% por Tópico', '% Top 10%', False),
        ('pct_top_1', '% Top 1% por Tópico', '% Top 1%', False),
        ('percentile', 'Percentil por Tópico', 'Percentil', False)
    ]
    
    col_name, title_suffix, y_label, is_production = metrics_map[tab_index]
    trends = data['topical_trends'].copy()
    
    if not show_all and 'top_topics' in data and not data['top_topics'].empty:
        top_names = data['top_topics'].head(10).index.tolist()
        trends = trends[trends['topic'].isin(top_names)]
    
    trends = trends[(trends[col_name].notnull()) & (trends['year'] <= 2025)].sort_values('year')
    if trends.empty: return
    
    fig = px.line(trends, x='year', y=col_name, color='topic',
                  title=f"{title_suffix} - {entity_name}",
                  labels={col_name: y_label, 'year': 'Año', 'topic': 'Tópico'},
                  markers=True, template="plotly_white")

    fig.update_layout(showlegend=not show_all, height=450, margin=dict(l=0, r=0, t=40, b=0))
    try:
        fig.update_xaxes(type='linear', tickformat='d')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error renderizando desglose por tópico: {e}")

def render_document_types(entity_name, df_types):
    """Renderiza la distribución de tipos documentales."""
    dist = get_type_distribution(df_types, entity_name)
    if dist is None or dist.empty:
        st.info(f"Sin datos de tipos documentales para {entity_name}")
        return

    dist['year'] = pd.to_numeric(dist['year'], errors='coerce')
    dist = dist[(dist['year'] >= 1950) & (dist['year'] <= 2025)].dropna(subset=['year'])
    dist['year'] = dist['year'].astype(int)
    
    dist = dist.pivot_table(index='year', columns='doc_type', values='count', aggfunc='sum').fillna(0)
    dist = dist.stack().reset_index(name='count')
    dist = dist.sort_values(['year', 'doc_type'])
    
    st.markdown(f"**📄 Análisis de Tipos Documentales: {entity_name}**")
    pie_data = dist.groupby('doc_type')['count'].sum().reset_index()
    fig_pie = px.pie(pie_data, values='count', names='doc_type', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pie.update_layout(showlegend=True, height=350, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

    fig = px.area(dist, x="year", y="count", color="doc_type",
                  title=f"Evolución por Tipo Documental: {entity_name}",
                  labels={"count": "Documentos", "year": "Año", "doc_type": "Tipo"},
                  template="plotly_white", height=400)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), xaxis=dict(range=[1950, 2025], type='linear'))
    fig.update_xaxes(tickformat='d')
    st.plotly_chart(fig, use_container_width=True)

def render_institution_types(entity_name, df_inst_types):
    """Renderiza la distribución de tipos de instituciones."""
    dist = get_inst_type_distribution(df_inst_types, entity_name)
    if dist is None or dist.empty:
        st.info(f"Sin datos de tipos de instituciones para {entity_name}")
        return

    dist['year'] = pd.to_numeric(dist['year'], errors='coerce')
    dist = dist[(dist['year'] >= 1950) & (dist['year'] <= 2025)].dropna(subset=['year'])
    dist['year'] = dist['year'].astype(int)
    
    dist = dist.pivot_table(index='year', columns='inst_type', values='count', aggfunc='sum').fillna(0)
    dist = dist.stack().reset_index(name='count')
    dist = dist.sort_values(['year', 'inst_type'])
    
    st.markdown(f"**🏢 Análisis de Sectores (Instituciones): {entity_name}**")
    pie_data = dist.groupby('inst_type')['count'].sum().reset_index()
    fig_pie = px.pie(pie_data, values='count', names='inst_type', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
    fig_pie.update_layout(showlegend=True, height=350, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

    fig = px.area(dist, x="year", y="count", color="inst_type",
                  title=f"Evolución por Sector: {entity_name}",
                  labels={"count": "Documentos", "year": "Año", "inst_type": "Sector"},
                  template="plotly_white", height=400)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), xaxis=dict(range=[1950, 2025], type='linear'))
    fig.update_xaxes(tickformat='d')
    st.plotly_chart(fig, use_container_width=True)

def render_entity_details(entity_name, data, df_types, df_inst_types, show_all=False):
    if not data: return

    # Diversidad Temática (Topics)
    with st.expander("🧩 Desglose de Tópicos Internos", expanded=True):
        if not data['top_topics'].empty:
            topics_to_show = data['top_topics'] if show_all else data['top_topics'].head(10)
            fig_topics = px.pie(values=topics_to_show.values, names=topics_to_show.index,
                               hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_topics.update_traces(textposition='inside', textinfo='label+percent')
            fig_topics.update_layout(showlegend=False, height=450, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_topics, use_container_width=True)
        else:
            st.info("Sin datos de tópicos.")

    # OA & Language
    m = data['metrics']
    col_oa, col_lang = st.columns(2)
    with col_oa:
        st.markdown("**Acceso Abierto**")
        oa_data = pd.DataFrame({
            'Tipo': ['Diamond', 'Gold', 'Green', 'Hybrid', 'Bronze', 'Closed'],
            'Valor': [m['pct_oa_diamond'], m['pct_oa_gold'], m['pct_oa_green'], 
                     m['pct_oa_hybrid'], m['pct_oa_bronze'], m['pct_oa_closed']]
        })
        fig_oa = px.bar(oa_data[oa_data['Valor']>0], x='Tipo', y='Valor', color='Tipo', color_discrete_sequence=px.colors.qualitative.Set3)
        fig_oa.update_layout(showlegend=False, height=300, xaxis_title=None, yaxis_title="%")
        st.plotly_chart(fig_oa, use_container_width=True)

    with col_lang:
        st.markdown("**Idiomas (Predominantes)**")
        l_data = pd.DataFrame({
            'Idioma': ['EN', 'ES', 'PT'],
            'Pct': [m['pct_lang_en'], m['pct_lang_es'], m['pct_lang_pt']]
        })
        fig_l = px.pie(l_data[l_data['Pct']>0], values='Pct', names='Idioma', color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'])
        fig_l.update_layout(showlegend=True, height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_l, use_container_width=True)

    # Document Types & Institution Types
    st.markdown("---")
    render_document_types(entity_name, df_types)
    st.markdown("---")
    render_institution_types(entity_name, df_inst_types)

def render_collaboration_network(df_collab):
    if df_collab is None or df_collab.empty:
        return
    
    # 1. Preparar la red con el Motor de Redes
    net = NetworkEngine()
    
    # Tomar top 100 relaciones para no saturar
    df_net = df_collab.sort_values('count', ascending=False).head(100)
    
    # Agregar nodos y aristas (el motor maneja clusters automáticamente)
    for _, row in df_net.iterrows():
        # Nodo A
        net.add_node(row['country_a'], label=row['country_a'], node_type="country")
        # Nodo B
        net.add_node(row['country_b'], label=row['country_b'], node_type="country")
        # Arista
        net.add_edge(row['country_a'], row['country_b'], weight=row['count'])
        
    # Calcular comunidades para colores
    net.compute_communities()
    
    # 2. Generar HTML de D3.js
    html_content = net.get_d3_html(title="Red de Colaboración Internacional")
    
    # 3. Renderizar en Streamlit
    components.html(html_content, height=650, scrolling=False)

def render_entity_institutions(entity_name, df_inst_all, period_mode, x_col, y_col, x_label, y_label):
    """Renderiza el análisis institucional para una entidad específica en formato de burbujas."""
    if df_inst_all is None or df_inst_all.empty:
        st.info("Sin datos institucionales.")
        return

    df_i = df_inst_all.copy()
    if entity_name == "Mundo": pass 
    elif entity_name == "México": df_i = df_i[df_i['country_code'] == 'MX']
    else: df_i = df_i[df_i['region'] == entity_name]

    if df_i.empty: return

    if period_mode == "Últimos 5 años (2021-2025)":
        df_i = df_i[(df_i['year'] >= 2021) & (df_i['year'] <= 2025)]

    df_calc = df_i.copy()
    metrics_to_weight = ['fwci', 'percentile', 'pct_top_10', 'pct_top_1']
    for m in metrics_to_weight: df_calc[f'{m}_prod'] = df_calc[m] * df_calc['doc_count']
    
    df_rank = df_calc.groupby(['institution_id', 'institution_name', 'country_code']).agg({
        'doc_count': 'sum', 'fwci_prod': 'sum', 'percentile_prod': 'sum', 'pct_top_10_prod': 'sum', 'pct_top_1_prod': 'sum', 'citations': 'sum'
    }).reset_index()
    
    for m in metrics_to_weight:
        df_rank[m] = df_rank[f'{m}_prod'] / df_rank['doc_count']
        df_rank[m] = df_rank[m].fillna(0)

    df_rank = df_rank.sort_values('doc_count', ascending=False).head(30)
    if df_rank.empty: return

    st.markdown(f"**🏢 Instituciones Líderes: {entity_name}**")
    df_rank['info'] = df_rank['institution_name'] + " (" + df_rank['country_code'] + ")"
    fig = px.scatter(df_rank, x=x_col, y=y_col, size="citations", hover_name="info",
                     title=f"Líderes en {entity_name}", labels={x_col: x_label, y_col: y_label, "citations": "Citas"},
                     template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True, key=f"inst_chart_{entity_name}")

# --- LAYOUT ---
entities = ["Mundo", "México"] + sorted(list(GLOBAL_REGIONS.keys()))
col_A, col_B, col_C = st.columns(3)
with col_A:
    ent1 = st.selectbox("Entidad A", entities, index=0)
    st.markdown(f"### 🌏 {ent1}")
with col_B:
    idx_latam = entities.index("Latinoamérica y Caribe") if "Latinoamérica y Caribe" in entities else 0
    ent2 = st.selectbox("Entidad B", entities, index=idx_latam)
    st.markdown(f"### 📍 {ent2}")
with col_C:
    idx_mex = entities.index("México") if "México" in entities else 0
    ent3 = st.selectbox("Entidad C", entities, index=idx_mex)
    st.markdown(f"### 🇲🇽 {ent3}")

ck1, ck2, ck3 = st.columns(3)
with ck1: data1 = render_entity_kpis(ent1, df_data, period_mode)
with ck2: data2 = render_entity_kpis(ent2, df_data, period_mode)
with ck3: data3 = render_entity_kpis(ent3, df_data, period_mode)

st.markdown("#### Evolución Temporal Sincronizada")
tab_labels = ["📈 Producción", "💥 FWCI", "🏆 % Top 10%", "🌟 % Top 1%", "📊 Percentil"]
tabs = st.tabs(tab_labels)
for i, tab in enumerate(tabs):
    with tab:
        st.markdown(f"**Total {tab_labels[i]}**")
        tc1, tc2, tc3 = st.columns(3)
        with tc1: render_entity_charts_synced(ent1, data1, i)
        with tc2: render_entity_charts_synced(ent2, data2, i)
        with tc3: render_entity_charts_synced(ent3, data3, i)
        st.markdown("---")
        st.markdown(f"**Desglose por Tópicos: {tab_labels[i]}**")
        tt1, tt2, tt3 = st.columns(3)
        with tt1: render_topical_evolution(ent1, data1, i, show_all=show_all_topics)
        with tt2: render_topical_evolution(ent2, data2, i, show_all=show_all_topics)
        with tt3: render_topical_evolution(ent3, data3, i, show_all=show_all_topics)

# Configuración de Ejes para Instituciones (Sidebar)
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🏢 Gráficos de Instituciones")
    inst_x_label = st.selectbox("Eje X (Burbujas)", list(INST_METRICS.keys()), index=0)
    inst_y_label = st.selectbox("Eje Y (Burbujas)", list(INST_METRICS.keys()), index=1)
    ix_col = INST_METRICS[inst_x_label]
    iy_col = INST_METRICS[inst_y_label]

# Detalles Detallados (Pie charts, sectores, revistas)
st.markdown("---")
st.markdown("#### Detalles por Entidad y Líderes")
cd1, cd2, cd3 = st.columns(3)
with cd1:
    render_entity_details(ent1, data1, df_types, df_inst_types, show_all=show_all_topics)
    render_entity_institutions(ent1, df_inst, period_mode, ix_col, iy_col, inst_x_label, inst_y_label)
    st.markdown("**📚 Top Revistas: " + ent1 + "**")
    st.dataframe(df_journals_top.head(10), use_container_width=True, hide_index=True)
with cd2:
    render_entity_details(ent2, data2, df_types, df_inst_types, show_all=show_all_topics)
    render_entity_institutions(ent2, df_inst, period_mode, ix_col, iy_col, inst_x_label, inst_y_label)
    st.markdown("**📚 Top Revistas: " + ent2 + "**")
    st.dataframe(df_journals_top.head(10), use_container_width=True, hide_index=True)
with cd3:
    render_entity_details(ent3, data3, df_types, df_inst_types, show_all=show_all_topics)
    render_entity_institutions(ent3, df_inst, period_mode, ix_col, iy_col, inst_x_label, inst_y_label)
    st.markdown("**📚 Top Revistas: " + ent3 + "**")
    st.dataframe(df_journals_top.head(10), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Frentes de Investigación: función definida antes de usarse
# ---------------------------------------------------------------------------

def _render_fronts_tab(subfield_name: str):
    """
    Pestaña de Frentes de Investigación.
    Lanza el pipeline como subproceso independiente (no bloquea el dashboard).
    El botón queda deshabilitado mientras el proceso corre.
    """
    import subprocess
    import sys
    from pathlib import Path

    RUNNING_KEY  = "fronts_running"
    PID_KEY      = "fronts_pid"
    SUBFIELD_KEY = "fronts_subfield"
    LOG_KEY      = "fronts_log_path"

    sub_clean = subfield_name.strip().lower().replace(' ', '_')
    log_path  = Path(f"data/cache_fronts/{sub_clean}/pipeline.log")
    done_path = Path(f"data/cache_fronts/{sub_clean}/.done")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Detectar si hay un proceso activo entre re-runs de Streamlit
    is_running = st.session_state.get(RUNNING_KEY, False)
    stored_pid = st.session_state.get(PID_KEY, None)

    if is_running and stored_pid:
        try:
            import psutil
            proc_check = psutil.Process(stored_pid)
            if proc_check.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                is_running = False
                st.session_state[RUNNING_KEY] = False
        except Exception:
            if done_path.exists():
                is_running = False
                st.session_state[RUNNING_KEY] = False

    st.markdown("### 🔬 Pipeline de Frentes de Investigación")
    st.markdown("""
    Detecta **frentes de investigación** mediante tres métodos complementarios:
    - 🔗 **Estructural** — Acoplamiento bibliográfico + Leiden
    - 📄 **Semántico** — SPECTER2 → UMAP(30d) → HDBSCAN
    - 🕸️ **Topológico** — FastRP sobre grafo heterogéneo (igraph)
    """)

    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        if "fronts_mode" not in st.session_state: st.session_state.fronts_mode = "sliding"
        mode = st.selectbox(
            "Segmentación temporal",
            ["sliding", "vigintiles", "both"],
            key="fronts_mode",
            help="sliding = ventanas de 3 años (recomendado para análisis reciente)",
            disabled=is_running
        )
    with opt_col2:
        if "fronts_workers" not in st.session_state: st.session_state.fronts_workers = 4
        workers = st.number_input(
            "Workers paralelos",
            min_value=1, max_value=16, 
            key="fronts_workers",
            help="Un proceso por bin temporal. Recomendado: N_cores / 2.",
            disabled=is_running
        )
    with opt_col3:
        if "fronts_force_from" not in st.session_state: st.session_state.fronts_force_from = "ninguno"
        force_from = st.selectbox(
            "Forzar recálculo desde",
            ["ninguno", "windows", "citations", "structural",
             "umap", "semantic", "topological", "ami", "tracking"],
            key="fronts_force_from",
            help="'ninguno' = usa cache existente para todo lo posible.",
            disabled=is_running
        )
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        # Persistencia del estado del checkbox
        if "fronts_use_cpu" not in st.session_state:
            st.session_state.fronts_use_cpu = False
            
        use_cpu = st.checkbox(
            "Forzar uso de CPU", 
            key="fronts_use_cpu",
            help="Ignora la GPU (útil si hay conflictos de memoria con LM Studio o drivers antiguos).",
            disabled=is_running
        )
    btn_col, status_col = st.columns([1, 3])

    with btn_col:
        launch_btn = st.button(
            "🚀 Lanzar Pipeline" if not is_running else "⏳ Procesando...",
            disabled=is_running,
            type="primary",
            use_container_width=True,
            key="btn_launch_fronts"
        )

    with status_col:
        if is_running:
            st.warning(
                f"**El pipeline está corriendo en el servidor** para "
                f"*{st.session_state.get(SUBFIELD_KEY, subfield_name)}*.  "
                "Este proceso puede tardar **varias horas** dependiendo del subcampo y workers.  "
                "Puedes cerrar esta pestaña — el proceso continúa en el servidor."
            )
        elif done_path.exists():
            import datetime
            done_dt = datetime.datetime.fromtimestamp(
                done_path.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M")
            st.success(f"✅ Pipeline completado ({done_dt}). Resultados disponibles.")
        else:
            st.info("Selecciona las opciones y haz clic en **🚀 Lanzar Pipeline**.")

    if launch_btn and not is_running:
        force_arg = [] if force_from == "ninguno" else ["--force-from", force_from]
        cpu_arg = ["--cpu"] if use_cpu else []
        cmd = [
            sys.executable, "-m", "fronts.run_pipeline",
            "--step", "all",
            "--subfield", subfield_name,
            "--mode", mode,
            "--workers", str(int(workers)),
        ] + force_arg + cpu_arg

        log_path.unlink(missing_ok=True)
        done_path.unlink(missing_ok=True)
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
            text=True
        )
        st.session_state[RUNNING_KEY]  = True
        st.session_state[PID_KEY]      = proc.pid
        st.session_state[SUBFIELD_KEY] = subfield_name
        st.session_state[LOG_KEY]      = str(log_path)
        st.rerun()

    if log_path.exists() and log_path.stat().st_size > 0:
        st.markdown("#### 📋 Log del proceso")
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail  = lines[-40:] if len(lines) > 40 else lines
            st.code("\n".join(tail), language="", wrap_lines=False)
        except Exception as e:
            st.caption(f"(no se pudo leer el log: {e})")
        if is_running:
            time.sleep(0.5)
            st.caption("🔄 La página se actualiza cada 15 s mientras el proceso está activo.")
            st.markdown(
                """<meta http-equiv="refresh" content="15">""",
                unsafe_allow_html=True
            )

    result_path = Path(f"data/cache_fronts/{sub_clean}/fronts_result.parquet")
    if result_path.exists() and not is_running:
        st.markdown("---")
        st.markdown("#### 📊 Vista previa de resultados")
        try:
            df_fronts = pd.read_parquet(result_path)
            r1, r2, r3 = st.columns(3)
            n_papers   = len(df_fronts)
            n_bins     = df_fronts['bin_id'].nunique() if 'bin_id' in df_fronts.columns else 0
            n_clusters = 0
            for col in ['cluster_leiden', 'cluster_semantic', 'cluster_topological']:
                if col in df_fronts.columns:
                    n_clusters = max(n_clusters,
                                     df_fronts[col][df_fronts[col] != -1].nunique())
            r1.metric("Papers procesados", f"{n_papers:,}")
            r2.metric("Ventanas temporales", n_bins)
            r3.metric("Clusters detectados (máx)", n_clusters)
            cols_show = ['id', 'publication_year', 'bin_id']
            for c in ['cluster_leiden', 'cluster_semantic', 'cluster_topological']:
                if c in df_fronts.columns:
                    cols_show.append(c)
            st.dataframe(df_fronts[cols_show].head(50),
                         use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"No se pudo leer el archivo de resultados: {e}")


# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("---")
st.subheader("Tablas de Resumen General")
res = get_summary_tables(df_data)
df_countries, df_topics, df_countries_total, df_ct_annual, df_ct_full, df_ct_2125, df_topics_total = res

if df_countries is not None:
    tabs_list = [
        "🌎 Países (Anual)", "🌎 Países (Totales)", 
        "🧩 Tópicos (Anual)", "🧩 Tópicos (Totales)",
        "📚 Revistas (Anual)", "📅 Evolución Países-Tópicos",
        "📊 Totales 2021-2025", "📈 Totales Históricos",
        "🤝 Colaboración", "🏢 Instituciones", "🔬 Frentes de Investigación"
    ]
    all_tabs = st.tabs(tabs_list)
    (tab_sum_1, tab_sum_1b, tab_sum_2, tab_sum_2b, tab_sum_3, 
     tab_sum_4, tab_sum_5, tab_sum_6, tab_sum_7, tab_sum_8, tab_fronts) = all_tabs
    
    with tab_sum_1:
        st.subheader("Producción e Impacto por País y Año")
        st.dataframe(df_countries, use_container_width=True, hide_index=True)
        download_csv_button(df_countries, "Paises_Anual")
        
    with tab_sum_1b:
        st.subheader("Producción e Impacto por País (Total Histórico)")
        st.dataframe(df_countries_total, use_container_width=True, hide_index=True)
        download_csv_button(df_countries_total, "Paises_Totales")
        
    with tab_sum_2:
        st.subheader("Producción e Impacto por Tópico y Año (Mundial)")
        st.dataframe(df_topics, use_container_width=True, hide_index=True)
        download_csv_button(df_topics, "Topicos_Anual")

    with tab_sum_2b:
        st.subheader("Producción e Impacto por Tópico (Total Histórico Mundial)")
        st.dataframe(df_topics_total, use_container_width=True, hide_index=True)
        download_csv_button(df_topics_total, "Topicos_Totales")
        
    with tab_sum_3:
        st.subheader("Top 100 Revistas Líderes (Periodo 2021-2025)")
        st.dataframe(df_journals_top.head(100), use_container_width=True, hide_index=True)
        download_csv_button(df_journals_top, "Top_Revistas")

    with tab_sum_4:
        st.subheader("Evolución de Artículos por País y Tópico (Anual)")
        st.dataframe(df_ct_annual, use_container_width=True, hide_index=True)
        download_csv_button(df_ct_annual, "Evolucion_Anual")

    with tab_sum_5:
        st.subheader("Totales de Producción Temática: 2021-2025")
        st.info("Suma total de documentos por tópico para cada país/región en el periodo actual.")
        st.dataframe(df_ct_2125, use_container_width=True, hide_index=True)
        download_csv_button(df_ct_2125, "Totales_Recientes")

    with tab_sum_6:
        st.subheader("Totales de Producción Temática: Periodo Completo")
        st.info("Suma histórica acumulada de documentos por tópico para cada entidad.")
        st.dataframe(df_ct_full, use_container_width=True, hide_index=True)
        download_csv_button(df_ct_full, "Totales_Historicos")

    with tab_sum_7:
        st.subheader("🤝 Matriz de Colaboración Internacional")
        if df_collab is not None and not df_collab.empty:
            st.info("Esta tabla y grafo muestran el número de co-autorías detectadas entre pares de países para este subcampo.")
            
            st.markdown("### 🌐 Red de Colaboración (Top 100 relaciones)")
            render_collaboration_network(df_collab)
            
            st.markdown("### 📊 Datos Tabulares")
            st.dataframe(df_collab, use_container_width=True, hide_index=True)
            download_csv_button(df_collab, "Colaboración")
        else:
            st.warning("No hay datos de colaboración para este subcampo. Intenta 'Forzar Recálculo'.")
            
    with tab_sum_8:
        st.subheader("🏢 Análisis de Instituciones Líderes")
        if df_inst is not None and not df_inst.empty:
            inst_col1, inst_col2 = st.columns([1, 2])
            with inst_col1:
                inst_regions = ["Todas"] + sorted(df_inst['region'].unique().tolist())
                sel_inst_region = st.selectbox("Filtrar por Región (Resumen)", inst_regions)
            
            df_inst_view = df_inst.copy()
            if sel_inst_region != "Todas":
                df_inst_view = df_inst_view[df_inst_view['region'] == sel_inst_region]
            
            if period_mode == "Últimos 5 años (2021-2025)":
                df_inst_view = df_inst_view[(df_inst_view['year'] >= 2021) & (df_inst_view['year'] <= 2025)]
            
            df_calc = df_inst_view.copy()
            metrics_to_weight = ['fwci', 'percentile', 'pct_top_10', 'pct_top_1']
            for m in metrics_to_weight: df_calc[f'{m}_prod'] = df_calc[m] * df_calc['doc_count']

            df_inst_rank = df_calc.groupby(['institution_id', 'institution_name', 'country_code', 'region']).agg({
                'doc_count': 'sum', 'fwci_prod': 'sum', 'percentile_prod': 'sum', 'pct_top_10_prod': 'sum', 'pct_top_1_prod': 'sum',
                'citations': 'sum', 'intl_collab': 'sum', 'sdg_docs': 'sum'
            }).reset_index()

            for m in metrics_to_weight:
                df_inst_rank[m] = df_inst_rank[f'{m}_prod'] / df_inst_rank['doc_count']
                df_inst_rank[m] = df_inst_rank[m].fillna(0)

            df_inst_rank = df_inst_rank.sort_values('doc_count', ascending=False)
            
            # 1. Benchmarking Plot (Burbujas)
            st.markdown(f"#### 🚀 Benchmarking: {inst_x_label} vs {inst_y_label}")
            fig_inst = px.scatter(
                df_inst_rank.head(50), 
                x=ix_col, y=iy_col, 
                size="citations", color="region",
                hover_name="institution_name",
                labels={ix_col: inst_x_label, iy_col: inst_y_label, "region": "Región", "citations": "Citas"},
                template="plotly_dark",
                height=600
            )
            st.plotly_chart(fig_inst, use_container_width=True)
            
            # 2. Ranking Table
            st.markdown(f"#### 🏆 Ranking Top 100 ({sel_inst_region})")
            st.dataframe(df_inst_rank.head(100).rename(columns={
                'institution_name': 'Institución', 'country_code': 'País', 'doc_count': 'Documentos', 'fwci': 'FWCI', 
                'pct_top_10': '% Top 10%', 'citations': 'Citas', 'intl_collab': 'Colab. Intl', 'sdg_docs': 'ODS'
            })[['Institución', 'País', 'Documentos', 'FWCI', '% Top 10%', 'Citas', 'Colab. Intl', 'ODS']], 
            use_container_width=True, hide_index=True)
            
            download_csv_button(df_inst_rank, f"Instituciones_{sel_inst_region}")
        else:
            st.warning("No hay datos institucionales calculados.")

    with tab_fronts:
        _render_fronts_tab(selected_subfield)

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard optimizado vía works_flat")
