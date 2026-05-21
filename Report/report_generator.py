import os
import sys
import argparse
import html
import random
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv

# Ensure root paths are in sys.path
BASE_PATH = Path(__file__).parent.parent.absolute()
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))
if str(BASE_PATH / 'src') not in sys.path:
    sys.path.insert(0, str(BASE_PATH / 'src'))

# Import existing helpers
import viz_bibliometrics
from lib.llm_utils import LLMConfig
from pipeline_topic.data_processor import (
    load_subfield_data,
    load_collaboration_data,
    load_institutional_data,
    load_types_data,
    load_inst_types_data,
    get_type_distribution,
    get_inst_type_distribution,
    get_entity_metrics,
    get_summary_tables
)

load_dotenv()

# --- CONFIGURATION ---
CACHE_TEMAS_DIR = BASE_PATH / 'data' / 'cache_temas'

# --- LLM API CALL HANDLER ---
def query_llm_narrative(section_title, data_table_md, subfield, target_entity, focus_topic_prompt):
    """
    Submits structured data to the LLM to write a rigorous, cienciometric analysis narrative.
    Includes error tolerance to prevent script failures if the server is offline.
    """
    url = LLMConfig.get_auth_url()
    model = LLMConfig.get_model_name()
    api_key = LLMConfig.get_api_key()
    
    # Clean URL handling (add /v1/chat/completions to base URL)
    if not url.endswith("/"):
        url += "/"
    chat_url = f"{url}chat/completions"
    
    system_prompt = (
        "Eres un analista cienciométrico sénior de una prestigiosa academia de ciencias mundiales. "
        "Tu tarea es redactar interpretaciones críticas, analíticas y rigurosas de datos bibliométricos en español. "
        "Debes utilizar un tono formal, académico, objetivo y preciso. Evita adjetivos sensacionalistas, "
        "clichés de marketing o hipérboles. Cíñete estrictamente a las métricas reales y tendencias de los datos proporcionados. "
        "No alucines con números que no existan en la tabla."
    )
    
    user_prompt = f"""
### ESTUDIO CIENTOMÉTRICO: {subfield}
#### SECCIÓN: {section_title}
#### ENFOQUE ESTRATÉGICO: {target_entity}

A continuación se presentan los datos cuantitativos reales en formato Markdown:
```markdown
{data_table_md}
```

#### INSTRUCCIONES DE REDACCIÓN NARRATIVA:
{focus_topic_prompt}

Escribe entre 3 y 4 párrafos completos estructurados como un artículo académico listo para publicación en revistas indexadas como *Scientometrics*. Utiliza vocabulario especializado (ej. impacto citacional ponderado, internacionalización de la ciencia, asimetría de flujos de conocimiento, ciencia abierta).
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2, # Baja temperatura para evitar alucinaciones
        "max_tokens": 1000
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Petición HTTP con timeout corto de 15 segundos
        response = requests.post(chat_url, json=payload, headers=headers, verify=False, timeout=20)
        if response.status_code == 200:
            res_json = response.json()
            return res_json['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"HTTP Status {response.status_code}: {response.text}")
    except Exception as e:
        # Fallback offline premium para no romper el flujo
        return f"""
<div class="ia-fallback-box">
    <p class="fallback-title">⚠️ Interpretación Crítica Pendiente de Conexión IA</p>
    <p>La síntesis analítica y discusión cualitativa para la sección <strong>{section_title}</strong> está temporalmente en espera. El motor cienciométrico local no pudo conectarse con el servidor de lenguaje cienciométrico en la nube (LM Studio) en este momento (Detalle: <em>{str(e)}</em>).</p>
    <p>Los indicadores de desempeño (KPI Cards), las matrices de datos y las visualizaciones interactivas de Plotly y la Red de Coautorías PyVis mostrados arriba reflejan las métricas cuantitativas completas y validadas en tiempo real.</p>
</div>
"""

# --- AUXILIARY MATHEMATICS ---
def calculate_cagr(df_data, entity_name):
    """Calcula la Tasa de Crecimiento Anual Compuesto (CAGR) de la producción científica."""
    if df_data is None or df_data.empty:
        return 0.0
    entity_type = 'Mundo' if entity_name == 'Mundo' else ('Country' if entity_name == 'México' else 'Region')
    lookup_name = 'Mundo' if entity_name == 'Mundo' else ('MX' if entity_name == 'México' else entity_name)
    
    df_ent = df_data[(df_data['entity_type'] == entity_type) & (df_data['entity_name'] == lookup_name)].copy()
    if df_ent.empty:
        return 0.0
        
    df_annual = df_ent.groupby('year')['doc_count'].sum().reset_index().sort_values('year')
    df_annual = df_annual[(df_annual['year'] >= 2010) & (df_annual['year'] <= 2025) & (df_annual['doc_count'] > 0)]
    
    if len(df_annual) < 2:
        return 0.0
        
    first_row = df_annual.iloc[0]
    last_row = df_annual.iloc[-1]
    
    n_years = last_row['year'] - first_row['year']
    if n_years <= 0:
        return 0.0
        
    cagr = (last_row['doc_count'] / first_row['doc_count']) ** (1 / n_years) - 1
    return cagr * 100

def df_to_markdown(df, index=False):
    """Converts a pandas DataFrame to a markdown table string without depending on 'tabulate'."""
    if df is None or df.empty:
        return ""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    headers = "| " + " | ".join(cols) + " |"
    separators = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join([str(val) for val in row]) + " |"
        rows.append(row_str)
    return "\n".join([headers, separators] + rows)

# --- MAIN GENERATOR FUNCTION ---
def generate_report(subfield, target_entity="México", suffix="", output_path=None):
    sub_clean = subfield.strip().replace(' ', '_').lower()
    
    # 1. CARGA DE DATOS DESDE LOS CACHÉS
    df_data = load_subfield_data(subfield, suffix=suffix)
    df_collab = load_collaboration_data(subfield, suffix=suffix)
    df_inst = load_institutional_data(subfield, suffix=suffix)
    df_types = load_types_data(subfield, suffix=suffix)
    df_inst_types = load_inst_types_data(subfield, suffix=suffix)
    
    cache_jr = CACHE_TEMAS_DIR / f"{sub_clean}_journals{suffix}.parquet"
    if not cache_jr.exists():
        cache_jr = CACHE_TEMAS_DIR / f"{sub_clean}_journals.parquet"
    df_journals = pd.read_parquet(cache_jr) if cache_jr.exists() else pd.DataFrame(columns=['Revista', 'URL', 'Artículos'])

    if df_data is None:
        raise FileNotFoundError(f"No se encontró el archivo principal de caché para el subcampo '{subfield}'. Asegúrate de calcularlo en el Dashboard primero.")

    is_global_mode = (target_entity == "Global (Comparativo Multiescala)")
    entity_for_kpis = "México" if is_global_mode else target_entity

    # 2. DEFINIR NOMBRE DE SALIDA
    if output_path is None:
        ent_clean = target_entity.strip().replace(' ', '_').lower()
        output_path = BASE_PATH / 'Report' / f"reporte_{sub_clean}_{ent_clean}{suffix}.html"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. EXTRACCIÓN DE MÉTRICAS GLOBALES (KPIs)
    world_metrics = get_entity_metrics(df_data, "Mundo", "Periodo Completo")
    entity_metrics = get_entity_metrics(df_data, entity_for_kpis, "Últimos 5 años (2021-2025)")
    
    if world_metrics is None or not world_metrics.get('metrics'):
        # Fallback si falla
        world_kpis = {'docs': 0, 'fwci': 0.0, 'top_10': 0.0, 'top_1': 0.0, 'percentile': 0.0}
    else:
        world_kpis = world_metrics['metrics']
        
    if entity_metrics is None or not entity_metrics.get('metrics'):
        ent_kpis = {'docs': 0, 'fwci': 0.0, 'top_10': 0.0, 'top_1': 0.0, 'percentile': 0.0, 'share': 0.0}
    else:
        ent_kpis = entity_metrics['metrics']
        
    cagr_world = calculate_cagr(df_data, "Mundo")
    cagr_ent = calculate_cagr(df_data, entity_for_kpis)

    # --- RENDERIZACIÓN DE GRÁFICOS A HTML CDN ---
    
    # Gráfica 1: Share Temporal por Región (Sección 2)
    df_reg_all = df_data[df_data['entity_type'] == 'Region'].copy()
    df_reg_agg = df_reg_all.groupby(['year', 'entity_name'])['doc_count'].sum().reset_index()
    # Calcular el total por año para sacar el share
    year_totals = df_data[df_data['entity_type'] == 'Mundo'].groupby('year')['doc_count'].sum().rename('world_total')
    df_reg_agg = df_reg_agg.merge(year_totals, on='year', how='left')
    df_reg_agg['share'] = np.where(df_reg_agg['world_total'] > 0, (df_reg_agg['doc_count'] / df_reg_agg['world_total']) * 100, 0)
    df_reg_agg = df_reg_agg[(df_reg_agg['year'] >= 2010) & (df_reg_agg['year'] <= 2025)].sort_values('year')
    
    fig_share = px.line(
        df_reg_agg, x='year', y='share', color='entity_name',
        title="Evolución de la Participación Mundial en la Producción Científica (%)",
        labels={'share': 'Share (%)', 'year': 'Año', 'entity_name': 'Región'},
        markers=True, template="plotly_white", height=380
    )
    fig_share.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    html_share = fig_share.to_html(include_plotlyjs='cdn', full_html=False)

    # Gráfica 2: Cuadrante Geopolítico por Regiones (Sección 2)
    fig_quad = viz_bibliometrics.render_geopolitical_quadrants(df_data, "Últimos 5 años (2021-2025)")
    html_quad = fig_quad.to_html(include_plotlyjs='cdn', full_html=False) if fig_quad else "<p class='no-data'>Métricas insuficientes para el cuadrante geopolítico</p>"

    # Gráfica 3: Red PyVis Coautoría (Sección 3)
    if df_collab is not None and not df_collab.empty:
        html_pyvis_raw = viz_bibliometrics.render_pyvis_network(df_collab, limit=50)
        # Escapado seguro para incrustar en el iframe
        escaped_pyvis = html.escape(html_pyvis_raw)
        html_pyvis_iframe = f'<iframe srcdoc="{escaped_pyvis}" style="width:100%; height:480px; border:1px solid #e2e8f0; border-radius: 8px; overflow:hidden;"></iframe>'
    else:
        html_pyvis_iframe = "<p class='no-data'>No hay lazos de colaboración registrados para este subcampo</p>"

    # Gráfica 4: Composición Taxonómica (Sunburst - Sección 4)
    # Extraer el domain y field desde el caché
    selected_domain = "Cached Domain"
    selected_field = "Cached Field"
    if df_data is not None and not df_data.empty:
        # Obtener jerarquía de fallback si está disponible
        from pipeline_topic.compute_metrics import get_hierarchy
        try:
            df_hier = get_hierarchy()
            sub_row = df_hier[df_hier['subfield'].str.lower() == subfield.lower()]
            if not sub_row.empty:
                selected_domain = sub_row.iloc[0]['domain']
                selected_field = sub_row.iloc[0]['field']
        except Exception:
            pass

    fig_sun = viz_bibliometrics.render_sunburst_hierarchy(df_data, "Mundo" if is_global_mode else target_entity, selected_domain, selected_field, subfield)
    html_sunburst = fig_sun.to_html(include_plotlyjs='cdn', full_html=False) if fig_sun else "<p class='no-data'>Datos insuficientes para el Sol Radiante taxonómico</p>"

    # Gráfica 5: Evolución de Sectores Institucionales (Sección 5)
    html_sectors = "<p class='no-data'>Métricas de distribución por sector no disponibles</p>"
    if df_inst_types is not None and not df_inst_types.empty:
        dist_sec = get_inst_type_distribution(df_inst_types, "Mundo" if is_global_mode else target_entity)
        if dist_sec is not None and not dist_sec.empty:
            dist_sec['year'] = pd.to_numeric(dist_sec['year'], errors='coerce')
            dist_sec = dist_sec[(dist_sec['year'] >= 2010) & (dist_sec['year'] <= 2025)].dropna(subset=['year'])
            dist_sec['year'] = dist_sec['year'].astype(int)
            dist_sec = dist_sec.pivot_table(index='year', columns='inst_type', values='count', aggfunc='sum').fillna(0)
            dist_sec = dist_sec.stack().reset_index(name='count').sort_values(['inst_type', 'year'])
            
            fig_sec = px.area(
                dist_sec, x="year", y="count", color="inst_type",
                title=f"Distribución del Volumen Científico por Sector Institucional ({'Mundo' if is_global_mode else target_entity})",
                labels={"count": "Artículos", "year": "Año", "inst_type": "Sector"},
                template="plotly_white", height=350,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_sec.update_layout(margin=dict(l=40, r=40, t=50, b=40))
            html_sectors = fig_sec.to_html(include_plotlyjs='cdn', full_html=False)

    # Gráfica 6: Evolución del Acceso Abierto (Sección 6)
    fig_oa_ev = viz_bibliometrics.render_oa_evolution(df_data, "Mundo" if is_global_mode else target_entity)
    html_oa_ev = fig_oa_ev.to_html(include_plotlyjs='cdn', full_html=False) if fig_oa_ev else "<p class='no-data'>Métricas de ciencia abierta no disponibles</p>"

    # Gráfica 7: Distribución ODS Institucional (Sección 7)
    fig_sdg = viz_bibliometrics.render_sdg_contributions(df_inst, "Mundo" if is_global_mode else target_entity, "Últimos 5 años (2021-2025)")
    html_sdg = fig_sdg.to_html(include_plotlyjs='cdn', full_html=False) if fig_sdg else "<p class='no-data'>No se detectaron contribuciones registradas a ODS en las instituciones líderes</p>"

    # --- TABLAS DE DATOS EN FORMATO HTML ---
    
    # Tabla 1: Top Países
    res_tables = get_summary_tables(df_data)
    df_countries = res_tables[2] # df_countries_total
    if df_countries is not None and not df_countries.empty:
        df_c_view = df_countries.sort_values('Documentos', ascending=False).head(10).copy()
        df_c_view['FWCI'] = df_c_view['FWCI'].map('{:.2f}'.format)
        df_c_view['% Top 10%'] = df_c_view['% Top 10%'].map('{:.1f}%'.format)
        df_c_view['Percentil'] = df_c_view['Percentil'].map('{:.1f}'.format)
        df_c_view['Documentos'] = df_c_view['Documentos'].map('{:,}'.format)
        html_table_countries = df_c_view.to_html(index=False, classes="table-premium")
    else:
        html_table_countries = "<p class='no-data'>Tabla de países no disponible</p>"

    # Tabla 2: Top Revistas (Sección 6)
    if df_journals is not None and not df_journals.empty:
        df_j_view = df_journals.head(10).copy()
        
        # Encontrar columna de artículos de forma de resiliencia
        art_col = None
        for col in df_j_view.columns:
            if col.lower() in ['artículos', 'articulos', 'doc_count', 'count', 'doc_cnt', 'articles']:
                art_col = col
                break
                
        if art_col:
            df_j_view[art_col] = pd.to_numeric(df_j_view[art_col], errors='coerce').fillna(0).astype(int)
            df_j_view['Artículos'] = df_j_view[art_col].map('{:,}'.format)
            if art_col != 'Artículos':
                df_j_view = df_j_view.drop(columns=[art_col])
        else:
            df_j_view['Artículos'] = "N/A"
            
        if 'URL' in df_j_view.columns:
            df_j_view['Revista'] = df_j_view.apply(
                lambda r: f'<a href="{r["URL"]}" target="_blank" class="journal-link">{r["Revista"]}</a>' if r["URL"] else r["Revista"],
                axis=1
            )
            df_j_view = df_j_view.drop(columns=['URL'])
            
        # Asegurar orden de columnas limpio
        cols_to_keep = [c for c in ['Revista', 'Artículos'] if c in df_j_view.columns]
        df_j_view = df_j_view[cols_to_keep]
        
        html_table_journals = df_j_view.to_html(index=False, escape=False, classes="table-premium")
    else:
        html_table_journals = "<p class='no-data'>Tabla de revistas no disponible</p>"

    # Tabla 3: Ranking de Instituciones (Sección 5 - Multi-segmento)
    html_inst_tables = ""
    if df_inst is not None and not df_inst.empty:
        # Filtrar por últimos 5 años para el ranking
        df_i_recent = df_inst[(df_inst['year'] >= 2021) & (df_inst['year'] <= 2025)].copy()
        if df_i_recent.empty:
            df_i_recent = df_inst.copy()
            
        df_calc = df_i_recent.copy()
        metrics_to_weight = ['fwci', 'percentile', 'pct_top_10', 'pct_top_1']
        for m in metrics_to_weight:
            df_calc[f'{m}_prod'] = df_calc[m] * df_calc['doc_count']

        df_rank_all = df_calc.groupby(['institution_id', 'institution_name', 'country_code', 'region', 'segment']).agg({
            'doc_count': 'sum', 'fwci_prod': 'sum', 'percentile_prod': 'sum', 'pct_top_10_prod': 'sum', 'pct_top_1_prod': 'sum', 'citations': 'sum'
        }).reset_index()

        for m in metrics_to_weight:
            df_rank_all[m] = df_rank_all[f'{m}_prod'] / df_rank_all['doc_count']
            df_rank_all[m] = df_rank_all[m].fillna(0)

        # Determinar la región de interés para el segmento regional
        from src.regions import get_region_for_country, GLOBAL_REGIONS
        if target_entity in GLOBAL_REGIONS:
            region_of_interest = target_entity
        elif target_entity == "México":
            region_of_interest = "Latinoamérica y Caribe"
        elif is_global_mode:
            region_of_interest = "Latinoamérica y Caribe"
        else:
            region_of_interest = "Latinoamérica y Caribe"

        df_regional_leaders = df_rank_all[
            (df_rank_all['segment'] == 'Regional') & (df_rank_all['region'] == region_of_interest)
        ].sort_values('doc_count', ascending=False).head(8)

        # Generar las tres tablas segmentadas (Global, Regional, México)
        segments_data = [
            ("Líderes Globales", df_rank_all.sort_values('doc_count', ascending=False).head(8)),
            (f"Líderes Regionales ({region_of_interest})", df_regional_leaders),
            ("Líderes de México (MX)", df_rank_all[df_rank_all['country_code'] == 'MX'].sort_values('doc_count', ascending=False).head(8))
        ]
        
        for title, df_seg in segments_data:
            if not df_seg.empty:
                df_view = df_seg[['institution_name', 'country_code', 'doc_count', 'fwci', 'pct_top_10', 'citations']].copy()
                df_view = df_view.rename(columns={
                    'institution_name': 'Institución', 'country_code': 'País',
                    'doc_count': 'Documentos', 'fwci': 'FWCI', 'pct_top_10': '% Top 10%', 'citations': 'Citas'
                })
                df_view['FWCI'] = df_view['FWCI'].map('{:.2f}'.format)
                df_view['% Top 10%'] = df_view['% Top 10%'].map('{:.1f}%'.format)
                df_view['Documentos'] = df_view['Documentos'].map('{:,}'.format)
                df_view['Citas'] = df_view['Citas'].map('{:,}'.format)
                
                html_inst_tables += f"""
                <div class="inst-segment-block">
                    <h5 class="segment-title">➔ {title}</h5>
                    {df_view.to_html(index=False, classes="table-premium")}
                </div>
                """
    if not html_inst_tables:
        html_inst_tables = "<p class='no-data'>Métricas institucionales segmentadas no disponibles</p>"

    # --- SÍNTESIS DE NARRATIVA NARRADA POR LLM ---
    print(f"[REPORT ENGINE] Iniciando redacción del estudio analítico vía LLM para '{subfield}'...")
    
    # Sección 1: Diagnóstico
    if is_global_mode:
        df_reg_recent = df_data[(df_data['entity_type'] == 'Region') & (df_data['year'] >= 2021) & (df_data['year'] <= 2025)].copy()
        df_reg_summary = df_reg_recent.groupby('entity_name').agg({
            'doc_count': 'sum',
            'fwci_sum': 'sum',
            'top_10_sum': 'sum'
        }).reset_index()
        df_reg_summary['fwci'] = (df_reg_summary['fwci_sum'] / df_reg_summary['doc_count']).round(2)
        df_reg_summary['top_10_pct'] = ((df_reg_summary['top_10_sum'] / df_reg_summary['doc_count']) * 100).round(1)
        df_reg_summary = df_reg_summary.rename(columns={'entity_name': 'Entidad', 'doc_count': 'Documentos', 'fwci': 'FWCI', 'top_10_pct': '% Top 10%'})
        
        df_mx_recent = df_data[(df_data['entity_type'] == 'Country') & (df_data['entity_name'] == 'MX') & (df_data['year'] >= 2021) & (df_data['year'] <= 2025)].copy()
        if not df_mx_recent.empty:
            mx_docs = df_mx_recent['doc_count'].sum()
            mx_fwci = (df_mx_recent['fwci_sum'].sum() / mx_docs).round(2) if mx_docs > 0 else 0.0
            mx_top10 = ((df_mx_recent['top_10_sum'].sum() / mx_docs) * 100).round(1) if mx_docs > 0 else 0.0
            df_mx_row = pd.DataFrame([{'Entidad': 'México', 'Documentos': mx_docs, 'FWCI': mx_fwci, '% Top 10%': mx_top10}])
            df_reg_summary = pd.concat([df_reg_summary, df_mx_row], ignore_index=True)
            
        df_wd_recent = df_data[(df_data['entity_type'] == 'Mundo') & (df_data['year'] >= 2021) & (df_data['year'] <= 2025)].copy()
        if not df_wd_recent.empty:
            wd_docs = df_wd_recent['doc_count'].sum()
            wd_fwci = (df_wd_recent['fwci_sum'].sum() / wd_docs).round(2) if wd_docs > 0 else 0.0
            wd_top10 = ((df_wd_recent['top_10_sum'].sum() / wd_docs) * 100).round(1) if wd_docs > 0 else 0.0
            df_wd_row = pd.DataFrame([{'Entidad': 'Mundo (Global)', 'Documentos': wd_docs, 'FWCI': wd_fwci, '% Top 10%': wd_top10}])
            df_reg_summary = pd.concat([df_reg_summary, df_wd_row], ignore_index=True)
            
        df_reg_summary = df_reg_summary.sort_values('Documentos', ascending=False)
        t1_md = df_to_markdown(df_reg_summary)
        
        s1_prompt = (
            "Analiza de manera integrada la producción científica global en el subcampo, comparando "
            "directamente el desempeño de las grandes regiones (Norte Global, Sur Global, China) y "
            "situando críticamente a México en este mapa mundial. Evalúa la madurez del subcampo "
            "a nivel macro, meso y micro, discutiendo el dinamismo y tasa de crecimiento del conocimiento."
        )
    else:
        t1_md = f"| Métrica | Global (Mundo) | Enfoque ({target_entity}) |\n| --- | --- | --- |\n"
        t1_md += f"| Documentos Publicados (Periodo) | {world_kpis['docs']:,} | {ent_kpis['docs']:,} |\n"
        t1_md += f"| CAGR % Crecimiento Anual | {cagr_world:.2f}% | {cagr_ent:.2f}% |\n"
        t1_md += f"| FWCI Promedio del Periodo | {world_kpis['fwci']:.2f} | {ent_kpis['fwci']:.2f} |\n"
        t1_md += f"| % Artículos en Top 10% Excelencia | {world_kpis['top_10']:.1f}% | {ent_kpis['top_10']:.1f}% |\n"
        t1_md += f"| % Artículos en Acceso Abierto (OA) | {world_kpis.get('pct_oa_gold', 0)+world_kpis.get('pct_oa_diamond', 0)+world_kpis.get('pct_oa_green', 0):.1f}% | {ent_kpis.get('pct_oa_gold', 0)+ent_kpis.get('pct_oa_diamond', 0)+ent_kpis.get('pct_oa_green', 0):.1f}% |\n"
        
        s1_prompt = (
            "Analiza el volumen de la producción, la tasa de crecimiento anual (CAGR) y los indicadores de impacto y excelencia "
            "del subcampo científico a nivel global, contrastándolo con el desempeño del país de enfoque. Evalúa si el subcampo "
            "se encuentra en una fase expansiva o de consolidación en base al CAGR."
        )
        
    narrative_sec1 = query_llm_narrative(
        "Diagnóstico Global del Subcampo", t1_md, subfield, target_entity, s1_prompt
    )

    # Sección 2: Geopolítica
    df_reg_all_recent = df_reg_all[(df_reg_all['year'] >= 2021) & (df_reg_all['year'] <= 2025)].copy()
    df_reg_calc = df_reg_all_recent.groupby('entity_name').agg({'doc_count': 'sum', 'fwci_sum': 'sum', 'top_10_sum': 'sum'}).reset_index()
    df_reg_calc['fwci'] = df_reg_calc['fwci_sum'] / df_reg_calc['doc_count']
    df_reg_calc['top_10_pct'] = (df_reg_calc['top_10_sum'] / df_reg_calc['doc_count']) * 100
    t2_md = df_to_markdown(df_reg_calc[['entity_name', 'doc_count', 'fwci', 'top_10_pct']])
    
    if is_global_mode:
        s2_prompt = (
            "Interpreta la dinámica geopolítica global y la evolución longitudinal de la cuota de producción mundial (Share %) "
            "de cada bloque regional. Discute la brecha cienciométrica entre el Norte Global y el Sur Global, analizando "
            "cómo la emergencia de China e India reconfigura el equilibrio del poder científico y qué posicionamiento guarda "
            "América Latina y el Caribe en esta estructura de poder."
        )
    else:
        s2_prompt = (
            "Analiza la distribución del poder científico mundial basándote en la producción de los bloques regionales y sus FWCI. "
            "Destaca el papel y balance entre el Norte Global (América del Norte y Europa) y el Sur Global (América Latina, África, Asia Emergente). "
            "Interpreta el gráfico de cuadrantes cienciométricos en base a estas diferencias."
        )
        
    narrative_sec2 = query_llm_narrative(
        "Dinámica Geopolítica y Participación Mundial", t2_md, subfield, target_entity, s2_prompt
    )

    # Sección 3: Benchmarking Países
    t3_md = ""
    if df_countries is not None:
        t3_md = df_to_markdown(df_countries.sort_values('Documentos', ascending=False).head(10))
        
    if is_global_mode:
        s3_prompt = (
            "Examina el benchmarking de los países líderes en el subcampo a nivel mundial y describe el rol estratégico de "
            "México dentro del contexto de América Latina y otros países emergentes. Discute la red de colaboración internacional "
            "(coautoría de PyVis) como un habilitador crítico de transferencia de conocimiento y ganancia de impacto (FWCI) "
            "para países del Sur Global."
        )
    else:
        s3_prompt = (
            f"Analiza la posición cienciométrica de los países líderes a nivel mundial y sitúa a {target_entity} en esta distribución. "
            "Examina cómo influye la colaboración internacional (red de coautoría co-ocurrente) en el impacto (FWCI) obtenido. "
            "Enfatiza la red de coautorías como catalizador de visibilidad científica."
        )
        
    narrative_sec3 = query_llm_narrative(
        "Benchmarking de Países y Colaboración Internacional", t3_md, subfield, target_entity, s3_prompt
    )

    # Sección 4: Identidad Temática
    t4_md = ""
    if df_data is not None:
        lookup_name = "Mundo" if is_global_mode else ('MX' if target_entity == 'México' else target_entity)
        top_t = df_data[df_data['entity_name'] == lookup_name]
        if not top_t.empty:
            t4_md = df_to_markdown(top_t.groupby('topic')['doc_count'].sum().sort_values(ascending=False).head(12).reset_index())
            
    if is_global_mode:
        s4_prompt = (
            "Analiza la composición jerárquica y estructura taxonómica del subcampo a nivel mundial. Identifica cuáles son los "
            "tópicos dominantes en la agenda global y contrástalos con la especialización o nichos de investigación que están "
            "adoptando las regiones en desarrollo, particularmente en América Latina."
        )
    else:
        s4_prompt = (
            f"Describe el perfil de especialización temática de {target_entity} en el subcampo. Identifica cuáles son los tópicos internos "
            "con mayor volumen y concentración y contrástalos con la agenda de investigación global. Utiliza el concepto cienciométrico "
            "de 'nicho científico' o especialización temática."
        )
        
    narrative_sec4 = query_llm_narrative(
        "Estructura e Identidad Temática", t4_md, subfield, target_entity, s4_prompt
    )

    # Sección 5: Paisaje Institucional
    t5_md = ""
    if df_inst is not None and not df_inst.empty:
        t5_md = df_to_markdown(df_inst[df_inst['country_code'] == 'MX'].groupby('institution_name')['doc_count'].sum().sort_values(ascending=False).head(10).reset_index())
        
    if is_global_mode:
        s5_prompt = (
            "Diagnostica la estructura del ecosistema institucional a nivel global y multisectorial (educación superior, corporativo, "
            "gobierno, salud). Discute cómo se distribuyen los liderazgos entre las instituciones de élite del Norte Global y "
            "la capacidad y madurez de las instituciones de América Latina y México para competir y colaborar en el subcampo."
        )
    else:
        s5_prompt = (
            f"Diagnostica la estructura del ecosistema institucional que investiga este tema en {target_entity}. Explica la importancia "
            "relativa del sector de Educación Superior (universidades públicas/privadas) en comparación con el sector salud (hospitales), "
            "gobierno y corporativo en base a las tablas segmentadas."
        )
        
    narrative_sec5 = query_llm_narrative(
        "Paisaje Institucional y Sectores", t5_md, subfield, target_entity, s5_prompt
    )

    # Sección 6: Ciencia Abierta
    t6_md = ""
    if df_journals is not None:
        t6_md = df_to_markdown(df_journals.head(10))
        
    if is_global_mode:
        s6_prompt = (
            "Analiza los hábitos y tendencias de publicación a nivel mundial, discutiendo la evolución de los modelos de Ciencia "
            "Abierta (Open Access). Compara críticamente la presión financiera que ejercen los cargos por procesamiento de artículos "
            "(APC) de las revistas del Norte Global frente a la relevancia de los repositorios y revistas Diamond/Green en "
            "el Sur Global."
        )
    else:
        s6_prompt = (
            "Analiza las vías de publicación y canales de difusión científica preferidos en este campo. Discute de manera crítica "
            "el balance entre el costo financiero de las publicaciones de pago (vías Híbridas y de pago APC) frente a la visibilidad y el "
            "desempeño cienciométrico de las vías gratuitas Diamond o Green, especialmente relevantes en el Sur Global."
        )
        
    narrative_sec6 = query_llm_narrative(
        "Vías de Difusión Científica y Ciencia Abierta", t6_md, subfield, target_entity, s6_prompt
    )

    # Sección 7: ODS
    t7_md = ""
    if df_inst is not None and 'sdg_docs' in df_inst.columns:
        t7_md = df_to_markdown(df_inst.groupby('institution_name')[['doc_count', 'sdg_docs']].sum().sort_values('sdg_docs', ascending=False).head(10).reset_index())
        
    if is_global_mode:
        s7_prompt = (
            "Evalúa la alineación social de la producción científica global con los Objetivos de Desarrollo Sostenible (ODS) de la ONU. "
            "Compara cómo las instituciones de diferentes bloques geopolíticos orientan su investigación hacia problemas globales "
            "(ej. salud, clima, energía limpia), destacando el valor social de la ciencia en América Latina y México."
        )
    else:
        s7_prompt = (
            f"Comenta la alineación social de la ciencia del subcampo con los Objetivos de Desarrollo Sostenible de la ONU en las "
            f"instituciones de {target_entity}. Explica cómo esta ciencia aplicada aporta valor social o respuestas críticas a "
            "desafíos socioambientales reales más allá de las métricas puras de citas académicas."
        )
        
    narrative_sec7 = query_llm_narrative(
        "Contribución al Desarrollo Sostenible (ODS)", t7_md, subfield, target_entity, s7_prompt
    )

    # --- COMPILACIÓN DE PLANTILLA HTML "JOURNAL STYLE" ---
    title_text = "Estudio Cientométrico Global y Multiescala" if is_global_mode else "Estudio Cientométrico Profundo y Benchmarking Multiescala"
    geopolitical_focus = "Comparativa Global Multiescala" if is_global_mode else (f"{target_entity} (LATAM)" if target_entity == "México" else target_entity)
    kpi_entity_label = "México" if is_global_mode else target_entity
    
    html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Estudio Cientométrico: {subfield}</title>
    <link href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        /* ESTILOS DE REVISTA CIENTÍFICA (JOURNAL EDITORIAL STYLE) */
        :root {{
            --primary: #1e3a8a;
            --primary-dark: #1e293b;
            --accent: #b91c1c;
            --text-dark: #334155;
            --text-muted: #64748b;
            --bg-page: #f8fafc;
            --bg-container: #ffffff;
            --border-color: #e2e8f0;
        }}

        body {{
            background-color: var(--bg-page);
            font-family: 'Outfit', sans-serif;
            color: var(--text-dark);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }}

        .journal-container {{
            max-width: 1100px;
            background-color: var(--bg-container);
            margin: 3rem auto;
            padding: 4rem;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
            border: 1px solid var(--border-color);
        }}

        /* Cabecera Editorial */
        .editorial-header {{
            border-bottom: 3px double var(--primary);
            padding-bottom: 2rem;
            margin-bottom: 3rem;
            text-align: center;
        }}

        .journal-meta {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .journal-title {{
            font-family: 'Merriweather', serif;
            font-size: 2.3rem;
            font-weight: 700;
            color: var(--primary-dark);
            margin: 0.5rem 0;
            line-height: 1.3;
        }}

        .journal-subtitle {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }}

        .editorial-meta-row {{
            display: flex;
            justify-content: center;
            gap: 40px;
            font-family: 'Outfit', sans-serif;
            font-size: 0.85rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            padding-top: 1rem;
            margin-top: 1rem;
        }}

        .meta-item strong {{
            color: var(--primary-dark);
        }}

        /* KPIs minimalistas de papel */
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 3.5rem;
        }}

        .kpi-card {{
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 18px;
            text-align: left;
            position: relative;
            transition: all 0.25s ease;
        }}

        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background-color: var(--primary);
            border-radius: 8px 0 0 8px;
        }}

        .kpi-label {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 6px;
        }}

        .kpi-val {{
            font-family: 'Merriweather', serif;
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--primary-dark);
        }}

        .kpi-sub {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.75rem;
            color: var(--accent);
            margin-top: 4px;
            font-weight: 500;
        }}

        /* Secciones Académicas */
        .academic-section {{
            margin-bottom: 4rem;
        }}

        .section-header-academic {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--primary-dark);
            border-bottom: 1px solid var(--primary);
            padding-bottom: 6px;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .section-num {{
            color: var(--accent);
            margin-right: 8px;
        }}

        /* Diseño de Doble Columna de Texto */
        .narrative-columns {{
            column-count: 2;
            column-gap: 2.5rem;
            column-rule: 1px solid var(--border-color);
            text-align: justify;
            margin-top: 1.5rem;
            margin-bottom: 2rem;
        }}

        .narrative-columns p {{
            margin-top: 0;
            margin-bottom: 1rem;
            line-height: 1.7;
            font-family: 'Merriweather', serif;
            font-size: 0.95rem;
            color: var(--text-dark);
        }}

        /* Letra Capital Premium */
        .narrative-columns p.intro::first-letter {{
            font-size: 2.6rem;
            font-weight: 700;
            float: left;
            line-height: 0.85;
            padding: 0.15rem 0.5rem 0 0;
            color: var(--primary);
            font-family: 'Outfit', sans-serif;
        }}

        /* Bloques de Gráficos e Iframe */
        .viz-block {{
            margin: 2rem 0;
            padding: 1rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background-color: #fafbfd;
        }}

        .viz-caption {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.8rem;
            color: var(--text-muted);
            text-align: center;
            margin-top: 8px;
            font-style: italic;
        }}

        /* Tablas Premium */
        .table-premium {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Outfit', sans-serif;
            font-size: 0.85rem;
            margin: 1.5rem 0;
            text-align: left;
        }}

        .table-premium th {{
            background-color: var(--primary-dark);
            color: #ffffff;
            font-weight: 600;
            padding: 10px 12px;
            border: 1px solid var(--primary-dark);
        }}

        .table-premium td {{
            padding: 9px 12px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-dark);
        }}

        .table-premium tr:nth-child(even) {{
            background-color: #fafafa;
        }}

        .journal-link {{
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
        }}

        .journal-link:hover {{
            text-decoration: underline;
            color: var(--accent);
        }}

        /* Segmentos Institucionales */
        .inst-segment-block {{
            margin-bottom: 2rem;
        }}

        .segment-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.95rem;
            color: var(--primary);
            margin-bottom: 8px;
            font-weight: 600;
        }}

        /* Caja de Fallback IA */
        .ia-fallback-box {{
            background-color: #fffbeb;
            border: 1px solid #fef3c7;
            border-left: 5px solid #d97706;
            border-radius: 8px;
            padding: 20px;
            margin: 1.5rem 0;
            font-family: 'Outfit', sans-serif;
            text-align: left;
        }}

        .fallback-title {{
            font-weight: 700;
            color: #b45309;
            margin-top: 0;
            margin-bottom: 8px;
        }}

        .ia-fallback-box p {{
            margin: 4px 0;
            font-size: 0.9rem;
            color: #78350f;
            line-height: 1.5;
        }}

        /* REGLAS DE IMPRESIÓN (PRINT-CSS PARA PDF Carta/A4) */
        @media print {{
            body {{
                background-color: #ffffff !important;
                color: #000000 !important;
            }}

            .journal-container {{
                box-shadow: none !important;
                border: none !important;
                padding: 0 !important;
                margin: 0 !important;
                max-width: 100% !important;
            }}

            .academic-section {{
                page-break-inside: avoid;
                page-break-after: auto;
                margin-bottom: 3rem !important;
            }}

            .viz-block {{
                page-break-inside: avoid;
                border: none !important;
                background: none !important;
            }}

            .table-premium {{
                page-break-inside: avoid;
            }}

            iframe {{
                page-break-inside: avoid;
            }}

            .ia-fallback-box {{
                border: 1px solid #ccc !important;
                background-color: #fff !important;
                color: #000 !important;
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>

    <div class="journal-container">
        
        <!-- CABECERA EDITORIAL -->
        <header class="editorial-header">
            <div class="journal-meta">Topics Analytica & Deep Research Engine ➔ Estudio Cientométrico</div>
            <h1 class="journal-title">{title_text}</h1>
            <div class="journal-subtitle">Subcampo: <strong>{subfield}</strong></div>
            
            <div class="editorial-meta-row">
                <div class="meta-item">Enfoque Geopolítico: <strong>{geopolitical_focus}</strong></div>
                <div class="meta-item">Fecha de Publicación: <strong>{pd.Timestamp.now().strftime("%d de %B, %Y")}</strong></div>
                <div class="meta-item">Mapeo de Base: <strong>OpenAlex Global Graph</strong></div>
            </div>
        </header>

        <!-- INDICADORES CLAVE (KPI CARDS) -->
        <section class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Volumen Global</div>
                <div class="kpi-val">{world_kpis['docs']:,}</div>
                <div class="kpi-sub">CAGR Histórico: {cagr_world:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Volumen {kpi_entity_label}</div>
                <div class="kpi-val">{ent_kpis['docs']:,}</div>
                <div class="kpi-sub">Share Mundial: {ent_kpis.get('share', 0):.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">FWCI {kpi_entity_label}</div>
                <div class="kpi-val">{ent_kpis['fwci']:.2f}</div>
                <div class="kpi-sub">Ref. Mundial: 1.00</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Excelencia {kpi_entity_label}</div>
                <div class="kpi-val">{ent_kpis['top_10']:.1f}%</div>
                <div class="kpi-sub">Top 10% Citados</div>
            </div>
        </section>

        <!-- SECCIÓN 1: DIAGNÓSTICO -->
        <section class="academic-section">
            <h3 class="section-header-academic">
                <span><span class="section-num">1.</span> Resumen Ejecutivo y Diagnóstico Global</span>
                <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Estadística Multiescala</span>
            </h3>
            <div class="narrative-columns">
                <p class="intro">{narrative_sec1.replace("<p class=\"intro\">", "").replace("</p>", "").replace("\n\n", "</p><p>")}</p>
            </div>
        </section>

        <!-- SECCIÓN 2: GEOPOLÍTICA -->
        <section class="academic-section">
            <h3 class="section-header-academic">
                <span><span class="section-num">2.</span> Dinámica Geopolítica y Participación Mundial (Share %)</span>
                <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Análisis de Bloques Científicos</span>
            </h3>
            
            <div class="viz-block">
                {html_share}
                <div class="viz-caption">Figura 2.1: Evolución longitudinal de la cuota de producción mundial (Share %) por regiones geográficas.</div>
            </div>

            <div class="viz-block">
                {html_quad}
                <div class="viz-caption">Figura 2.2: Cuadrante de Posicionamiento Geopolítico. FWCI (Eje Y) vs Volumen de Producción Reciente (Eje X) por regiones.</div>
            </div>

            <div class="narrative-columns">
                <p>{narrative_sec2.replace("\n\n", "</p><p>")}</p>
            </div>
        </section>

        <!-- SECCIÓN 3: BENCHMARKING PAÍSES -->
        <section class="academic-section">
            <h3 class="section-header-academic">
                <span><span class="section-num">3.</span> Benchmarking de Países y Red de Colaboración</span>
                <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Estructura de Coautoría</span>
            </h3>

            <div class="viz-block">
                <h4 style="font-family: 'Outfit', sans-serif; font-size: 0.95rem; margin-top: 0; color: var(--primary);">Tabla 3.1: Top 10 Países Líderes en Desempeño Cientométrico</h4>
                {html_table_countries}
            </div>

            <div class="viz-block">
                <h4 style="font-family: 'Outfit', sans-serif; font-size: 0.95rem; margin-top: 0; color: var(--primary);">Figura 3.1: Red Topológica y Comunidades de Coautoría Científica Internacional</h4>
                {html_pyvis_iframe}
                <div class="viz-caption">Gráfico interactivo de coautorías de PyVis. Los enlaces y nodos denotan la intensidad de co-publicaciones en el subcampo.</div>
            </div>

            <div class="narrative-columns">
                <p>{narrative_sec3.replace("\n\n", "</p><p>")}</p>
            </div>
        </section>

        <!-- SECCIÓN 4: IDENTIDAD TEMÁTICA -->
        <section class="academic-section">
            <h3 class="section-header-academic">
                <span><span class="section-num">4.</span> Estructura Taxonómica y Perfil Temático</span>
                <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Composición de Tópicos</span>
            </h3>

            <div class="viz-block">
                {html_sunburst}
                <div class="viz-caption">Figura 4.1: Sol Radiante (Sunburst) jerárquico. Distribución interna del subcampo en tópicos específicos ({"Mundo" if is_global_mode else target_entity}).</div>
            </div>

            <div class="narrative-columns">
                <p>{narrative_sec4.replace("\n\n", "</p><p>")}</p>
            </div>
        </section>

        <!-- SECCIÓN 5: ENTORNO INSTITUCIONAL -->
        <section class="academic-section">
            <h3 class="section-header-academic">
                <span><span class="section-num">5.</span> Paisaje Institucional y Estructura por Sectores</span>
                <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Ecosistema Universitario y de Salud</span>
            </h3>

            <div class="viz-block">
                {html_sectors}
                <div class="viz-caption">Figura 5.1: Evolución histórica de la producción clasificada por tipo de sector institucional en {"el Mundo" if is_global_mode else "la entidad"}.</div>
            </div>

            <div class="viz-block">
                <h4 style="font-family: 'Outfit', sans-serif; font-size: 0.95rem; margin-top: 0; color: var(--primary); margin-bottom: 1rem;">Mapeo Institucional Multi-Segmentado</h4>
                {html_inst_tables}
            </div>

            <div class="narrative-columns">
                <p>{narrative_sec5.replace("\n\n", "</p><p>")}</p>
            </div>
        </section>

        <!-- SECCIÓN 6: CIENCIA ABIERTA -->
        <section class="academic-section">
            <h3 class="section-header-academic">
                <span><span class="section-num">6.</span> Vías de Difusión y Acceso Abierto (Science Openness)</span>
                <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Canales Editoriales y APC</span>
            </h3>

            <div class="viz-block">
                {html_oa_ev}
                <div class="viz-caption">Figura 6.1: Evolución del modelo de publicación. Proporción de vías cerradas vs. abiertas a lo largo del periodo.</div>
            </div>

            <div class="viz-block">
                <h4 style="font-family: 'Outfit', sans-serif; font-size: 0.95rem; margin-top: 0; color: var(--primary);">Tabla 6.1: Top 10 Revistas y Canales Editoriales de Preferencia (2021-2025)</h4>
                {html_table_journals}
            </div>

            <div class="narrative-columns">
                <p>{narrative_sec6.replace("\n\n", "</p><p>")}</p>
            </div>
        </section>

        <!-- SECCIÓN 7: ODS -->
        <section class="academic-section" style="margin-bottom: 1rem;">
            <h3 class="section-header-academic">
                <span><span class="section-num">7.</span> Alineación con Objetivos de Desarrollo Sostenible (ODS)</span>
                <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Contribución y Valor Social</span>
            </h3>

            <div class="viz-block">
                {html_sdg}
                <div class="viz-caption">Figura 7.1: Volumen institucional de documentos alineados con ODS y ratio de alineación (%) por actor relevante.</div>
            </div>

            <div class="narrative-columns" style="margin-bottom: 1rem;">
                <p>{narrative_sec7.replace("\n\n", "</p><p>")}</p>
            </div>
        </section>

        <!-- FOOTER EDITORIAL -->
        <footer style="border-top: 1px solid var(--border-color); padding-top: 1.5rem; text-align: center; font-family: 'Outfit', sans-serif; font-size: 0.75rem; color: var(--text-muted);">
            <p>© {pd.Timestamp.now().year} Academia de Ciencias - Topics Analytical Engine. Reporte Auto-Contenido con Gráficos Dinámicos Plotly & PyVis.</p>
            <p style="font-size: 0.65rem;">Mapeos normalizados con base en esquemas cienciométricos de la OCDE y clasificaciones de OpenAlex de la ONU.</p>
        </footer>

    </div>

</body>
</html>
"""
    
    # 4. ESCRIBIR EL REPORTE EN EL ARCHIVO DE SALIDA
    output_path.write_text(html_template, encoding="utf-8")
    print(f"[REPORT ENGINE] Exito! Reporte escrito de forma autocontenida en: '{output_path.absolute()}'")
    return output_path

# --- COMMAND LINE INTERFACE (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de Reportes Cientométricos Avanzados (Deep Research)")
    parser.add_argument("--subfield", type=str, default="Pulmonary and Respiratory Medicine", help="Nombre del subcampo a analizar")
    parser.add_argument("--entity", type=str, default="México", help="País o bloque objetivo de enfoque")
    parser.add_argument("--suffix", type=str, default="", help="Sufijo para cargar el archivo Parquet de flat u otro")
    parser.add_argument("--output", type=str, default=None, help="Nombre del archivo HTML de salida")
    
    args = parser.parse_args()
    
    try:
        generate_report(args.subfield, args.entity, args.suffix, args.output)
    except Exception as e:
        print(f"[ERROR] Error en la generacion del reporte: {e}")
        sys.exit(1)
