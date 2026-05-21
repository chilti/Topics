import pandas as pd
import numpy as np
import plotly.express as px

# Diccionario completo de mapeo de ISO-2 (ClickHouse/OpenAlex) a ISO-3 (Plotly Choropleth)
ISO2_TO_ISO3 = {
    'AF': 'AFG', 'AX': 'ALA', 'AL': 'ALB', 'DZ': 'DZA', 'AS': 'ASM', 'AD': 'AND', 'AO': 'AGO', 'AI': 'AIA', 'AQ': 'ATA',
    'AG': 'ATG', 'AR': 'ARG', 'AM': 'ARM', 'AW': 'ABW', 'AU': 'AUS', 'AT': 'AUT', 'AZ': 'AZE', 'BS': 'BHS', 'BH': 'BHR',
    'BD': 'BGD', 'BB': 'BRB', 'BY': 'BLR', 'BE': 'BEL', 'BZ': 'BLZ', 'BJ': 'BEN', 'BM': 'BMU', 'BT': 'BTN', 'BO': 'BOL',
    'BQ': 'BES', 'BA': 'BIH', 'BW': 'BWA', 'BV': 'BVT', 'BR': 'BRA', 'IO': 'IOT', 'BN': 'BRN', 'BG': 'BGR', 'BF': 'BFA',
    'BI': 'BDI', 'CV': 'CPV', 'KH': 'KHM', 'CM': 'CMR', 'CA': 'CAN', 'KY': 'CYM', 'CF': 'CAF', 'TD': 'TCD', 'CL': 'CHL',
    'CN': 'CHN', 'CX': 'CXR', 'CC': 'CCK', 'CO': 'COL', 'KM': 'COM', 'CD': 'COD', 'CG': 'COG', 'CK': 'COK', 'CR': 'CRI',
    'CI': 'CIV', 'HR': 'HRV', 'CU': 'CUB', 'CW': 'CUW', 'CY': 'CYP', 'CZ': 'CZE', 'DK': 'DNK', 'DJ': 'DJI', 'DM': 'DMA',
    'DO': 'DOM', 'EC': 'ECU', 'EG': 'EGY', 'SV': 'SLV', 'GQ': 'GNQ', 'ER': 'ERI', 'EE': 'EST', 'SZ': 'SWZ', 'ET': 'ETH',
    'FK': 'FLK', 'FO': 'FRO', 'FJ': 'FJI', 'FI': 'FIN', 'FR': 'FRA', 'GF': 'GUF', 'PF': 'PYF', 'TF': 'ATF', 'GA': 'GAB',
    'GM': 'GMB', 'GE': 'GEO', 'DE': 'DEU', 'GH': 'GHA', 'GI': 'GIB', 'GR': 'GRC', 'GL': 'GRL', 'GD': 'GRD', 'GP': 'GLP',
    'GU': 'GUM', 'GT': 'GTM', 'GG': 'GGY', 'GN': 'GIN', 'GW': 'GNB', 'GY': 'GUY', 'HT': 'HTI', 'HM': 'HMD', 'VA': 'VAT',
    'HN': 'HND', 'HK': 'HKG', 'HU': 'HUN', 'IS': 'ISL', 'IN': 'IND', 'ID': 'IDN', 'IR': 'IRN', 'IQ': 'IRQ', 'IE': 'IRL',
    'IM': 'IMN', 'IL': 'ISR', 'IT': 'ITA', 'JM': 'JAM', 'JP': 'JPN', 'JE': 'JEY', 'JO': 'JOR', 'KZ': 'KAZ', 'KE': 'KEN',
    'KI': 'KIR', 'KP': 'PRK', 'KR': 'KOR', 'KW': 'KWT', 'KG': 'KGZ', 'LA': 'LAO', 'LV': 'LVA', 'LB': 'LBN', 'LS': 'LSO',
    'LR': 'LBR', 'LY': 'LBY', 'LI': 'LIE', 'LT': 'LTU', 'LU': 'LUX', 'MO': 'MAC', 'MG': 'MDG', 'MW': 'MWI', 'MY': 'MYS',
    'MV': 'MDV', 'ML': 'MLI', 'MT': 'MLT', 'MH': 'MHL', 'MQ': 'MTQ', 'MR': 'MRT', 'MU': 'MUS', 'YT': 'MYT', 'MX': 'MEX',
    'FM': 'FSM', 'MD': 'MDA', 'MC': 'MCO', 'MN': 'MNG', 'ME': 'MNE', 'MS': 'MSR', 'MA': 'MAR', 'MZ': 'MOZ', 'MM': 'MMR',
    'NA': 'NAM', 'NR': 'NRU', 'NP': 'NPL', 'NL': 'NLD', 'NC': 'NCL', 'NZ': 'NZL', 'NI': 'NIC', 'NE': 'NER', 'NG': 'NGA',
    'NU': 'NIU', 'NF': 'NFK', 'MP': 'MNP', 'NO': 'NOR', 'OM': 'OMN', 'PK': 'PAK', 'PW': 'PLW', 'PS': 'PSE', 'PA': 'PAN',
    'PG': 'PNG', 'PY': 'PRY', 'PE': 'PER', 'PH': 'PHL', 'PN': 'PCN', 'PL': 'POL', 'PT': 'PRT', 'PR': 'PRI', 'QA': 'QAT',
    'RE': 'REU', 'RO': 'ROU', 'RU': 'RUS', 'RW': 'RWA', 'BL': 'BLM', 'SH': 'SHN', 'KN': 'KNA', 'LC': 'LCA', 'MF': 'MAF',
    'PM': 'SPM', 'VC': 'VCT', 'WS': 'WSM', 'SM': 'SMR', 'ST': 'STP', 'SA': 'SAU', 'SN': 'SEN', 'RS': 'SRB', 'SC': 'SYC',
    'SL': 'SLE', 'SG': 'SGP', 'SX': 'SXM', 'SK': 'SVK', 'SI': 'SVN', 'SB': 'SLB', 'SO': 'SOM', 'ZA': 'ZAF', 'GS': 'SGS',
    'SS': 'SSD', 'ES': 'ESP', 'LK': 'LKA', 'SD': 'SDN', 'SR': 'SUR', 'SJ': 'SJM', 'SE': 'SWE', 'CH': 'CHE', 'SY': 'SYR',
    'TW': 'TWN', 'TJ': 'TJK', 'TZ': 'TZA', 'TH': 'THA', 'TL': 'TLS', 'TG': 'TGO', 'TK': 'TKL', 'TO': 'TON', 'TT': 'TTO',
    'TN': 'TUN', 'TR': 'TUR', 'TM': 'TKM', 'TC': 'TCA', 'TV': 'TUV', 'UG': 'UGA', 'UA': 'UKR', 'AE': 'ARE', 'GB': 'GBR',
    'UM': 'UMI', 'US': 'USA', 'UY': 'URY', 'UZ': 'UZB', 'VU': 'VUT', 'VE': 'VEN', 'VN': 'VNM', 'VG': 'VGB', 'VI': 'VIR',
    'WF': 'WLF', 'EH': 'ESH', 'YE': 'YEM', 'ZM': 'ZMB', 'ZW': 'ZWE'
}

# Nombres comunes en español de países para visualizaciones
COUNTRY_NAMES = {
    'AR': 'Argentina', 'AU': 'Australia', 'AT': 'Austria', 'BE': 'Bélgica', 'BR': 'Brasil', 'CA': 'Canadá', 
    'CL': 'Chile', 'CN': 'China', 'CO': 'Colombia', 'CR': 'Costa Rica', 'CU': 'Cuba', 'CZ': 'República Checa', 
    'DK': 'Dinamarca', 'EC': 'Ecuador', 'EG': 'Egipto', 'SV': 'El Salvador', 'ES': 'España', 'FI': 'Finlandia', 
    'FR': 'Francia', 'DE': 'Alemania', 'GR': 'Grecia', 'GT': 'Guatemala', 'HN': 'Honduras', 'IN': 'India', 
    'ID': 'Indonesia', 'IE': 'Irlanda', 'IL': 'Israel', 'IT': 'Italia', 'JP': 'Japón', 'MX': 'México', 
    'NL': 'Países Bajos', 'NZ': 'Nueva Zelanda', 'NO': 'Noruega', 'PK': 'Pakistán', 'PA': 'Panamá', 
    'PE': 'Perú', 'PL': 'Polonia', 'PT': 'Portugal', 'RU': 'Rusia', 'SA': 'Arabia Saudita', 'SG': 'Singapur', 
    'ZA': 'Sudáfrica', 'SE': 'Suecia', 'CH': 'Suiza', 'TR': 'Turquía', 'GB': 'Reino Unido', 'US': 'Estados Unidos', 
    'UY': 'Uruguay', 'VE': 'Venezuela'
}

# Paleta de colores consistente de Acceso Abierto
OA_COLORS = {
    'Diamond': '#60a5fa',  # Azul suave
    'Gold': '#fbbf24',     # Dorado/Amarillo suave
    'Green': '#34d399',    # Verde suave
    'Hybrid': '#c084fc',   # Púrpura suave
    'Bronze': '#f87171',   # Rojo/Rosa suave
    'Closed': '#94a3b8'    # Gris/Slate suave
}

def render_collaboration_map(df_collab, target_country_code):
    """
    Genera un mapa coroplético interactivo (Plotly) que muestra las coautorías 
    del país seleccionado con otros países a partir del archivo de colaboración.
    """
    if df_collab is None or df_collab.empty:
        return None

    target = target_country_code.strip().upper()

    df_a = df_collab[df_collab['country_a'] == target][['country_b', 'count']].rename(
        columns={'country_b': 'partner_iso2', 'count': 'collaborations'}
    )
    df_b = df_collab[df_collab['country_b'] == target][['country_a', 'count']].rename(
        columns={'country_a': 'partner_iso2', 'count': 'collaborations'}
    )

    df_target = pd.concat([df_a, df_b], ignore_index=True)

    if df_target.empty:
        return None

    df_target = df_target.groupby('partner_iso2')['collaborations'].sum().reset_index()
    df_target = df_target.sort_values('collaborations', ascending=False)
    df_target['iso_a3'] = df_target['partner_iso2'].map(ISO2_TO_ISO3)
    df_target['Pais'] = df_target['partner_iso2'].map(lambda x: COUNTRY_NAMES.get(x, x))
    df_target = df_target.dropna(subset=['iso_a3'])

    target_name = COUNTRY_NAMES.get(target, target)
    fig = px.choropleth(
        df_target,
        locations="iso_a3",
        color="collaborations",
        hover_name="Pais",
        hover_data={"iso_a3": False, "collaborations": True},
        color_continuous_scale="Blues",
        projection="natural earth",
        labels={"collaborations": "Coautorías"},
        title=f"Mapa Mundial de Alianzas Científicas de {target_name}"
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            landcolor="#f4f4f4",
            oceancolor="#eaf2f8",
            showocean=True,
            lakecolor="#eaf2f8",
            showlakes=True
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title="Coautorías",
            thicknessmode="pixels", thickness=15,
            lenmode="fraction", len=0.6,
            yanchor="middle", y=0.5,
            xanchor="left", x=0.02
        )
    )

    return fig

def render_pyvis_network(df_collab, limit=80):
    """
    Genera un grafo interactivo de red topológica de colaboración entre países usando PyVis.
    """
    from pyvis.network import Network
    from regions import get_region_for_country
    
    if df_collab is None or df_collab.empty:
        return ""
    
    # Tomar los top N enlaces para mantener la visualización limpia
    df_net = df_collab.head(limit)
    
    # Identificar todos los países únicos en esta sub-red
    countries = list(set(df_net['country_a'].tolist() + df_net['country_b'].tolist()))
    
    # Calcular el volumen total de colaboraciones por país (para el tamaño del nodo)
    node_weights = {}
    for c in countries:
        w_a = df_net[df_net['country_a'] == c]['count'].sum()
        w_b = df_net[df_net['country_b'] == c]['count'].sum()
        node_weights[c] = int(w_a + w_b)
        
    # Inicializar la Red de PyVis
    net = Network(height="450px", width="100%", bgcolor="#ffffff", font_color="#1e293b", notebook=False)
    
    # Colores por Región
    REGION_COLORS = {
        'China': '#fda4af',                  # Rosa/Rojo pastel
        'Asia Emergente': '#fed7aa',          # Naranja pastel
        'Latinoamérica y Caribe': '#86efac',   # Verde pastel
        'África Subsahariana': '#fef08a',      # Amarillo pastel
        'MENA': '#c7d2fe',                     # Indigo pastel
        'Norteamérica Anglosajona': '#93c5fd',  # Azul pastel
        'Europa Central/Occidental': '#c084fc',  # Púrpura pastel
        'Europa del Este': '#f472b6',          # Rosa pastel
        'Asia-Pacífico Desarrollado': '#5eead4',  # Teal pastel
        'Other': '#cbd5e1'                     # Gris pastel
    }
    
    # Agregar Nodos
    for c in countries:
        region = get_region_for_country(c)
        color = REGION_COLORS.get(region, REGION_COLORS['Other'])
        c_name = COUNTRY_NAMES.get(c, c)
        
        # El tamaño del nodo será proporcional a su peso (raíz de las colaboraciones)
        weight = node_weights.get(c, 5)
        size = 12 + int(np.sqrt(weight) * 2.5)
        
        # Tooltip flotante con diseño HTML limpio
        title_html = f"""
        <div style="font-family: sans-serif; padding: 6px; font-size: 12px; color: #1e293b; background: white; border: 1px solid #ccc; border-radius: 4px;">
            <b>{c_name}</b> ({c})<br/>
            Región: {region}<br/>
            Coautorías en Red: {weight}
        </div>
        """
        
        net.add_node(
            c,
            label=c_name,
            title=title_html,
            color=color,
            size=size,
            shape="dot",
            borderWidth=1,
            borderColor="#475569"
        )
        
    # Agregar Aristas (Edges)
    for _, row in df_net.iterrows():
        c_a = row['country_a']
        c_b = row['country_b']
        count = int(row['count'])
        
        # El grosor de la arista será proporcional al número de colaboraciones
        width = 1 + int(np.sqrt(count) / 1.5)
        
        net.add_edge(
            c_a,
            c_b,
            value=count,
            width=width,
            color="#cbd5e1",
            title=f"Coautorías: {count}"
        )
        
    # Configurar física y comportamiento (diseño de alta estabilidad y repulsión fluida)
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 150
        }
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true
      }
    }
    """)
    
    return net.generate_html()

def render_oa_donut(metrics_dict, title_prefix=""):
    """
    Genera un gráfico de dona (Pie Chart) premium para mostrar la distribución
    de las vías de Acceso Abierto.
    """
    if not metrics_dict:
        return None
        
    oa_data = pd.DataFrame({
        'Vía': ['Diamond', 'Gold', 'Green', 'Hybrid', 'Bronze', 'Closed'],
        'Porcentaje': [
            metrics_dict.get('pct_oa_diamond', 0),
            metrics_dict.get('pct_oa_gold', 0),
            metrics_dict.get('pct_oa_green', 0),
            metrics_dict.get('pct_oa_hybrid', 0),
            metrics_dict.get('pct_oa_bronze', 0),
            metrics_dict.get('pct_oa_closed', 0)
        ]
    })
    
    # Filtrar solo valores positivos para una dona limpia
    oa_data = oa_data[oa_data['Porcentaje'] > 0]
    
    if oa_data.empty:
        return None
        
    fig = px.pie(
        oa_data,
        values='Porcentaje',
        names='Vía',
        hole=0.4,
        color='Vía',
        color_discrete_map=OA_COLORS,
        title=f"Distribución de Acceso Abierto {title_prefix}",
        template="plotly_white",
        height=320
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    return fig

def render_oa_evolution(df_data, entity_name):
    """
    Genera un gráfico de áreas apiladas (px.area) que muestra la evolución
    temporal de las vías de Acceso Abierto para la entidad seleccionada.
    """
    if df_data is None or df_data.empty:
        return None

    entity_type = 'Mundo' if entity_name == 'Mundo' else ('Country' if entity_name == 'México' else 'Region')
    lookup_name = 'Mundo' if entity_name == 'Mundo' else ('MX' if entity_name == 'México' else entity_name)
    
    df_ent = df_data[(df_data['entity_type'] == entity_type) & (df_data['entity_name'] == lookup_name)].copy()
    if df_ent.empty:
        return None

    df_ent['year'] = pd.to_numeric(df_ent['year'], errors='coerce')
    df_ent = df_ent[(df_ent['year'] >= 2000) & (df_ent['year'] <= 2025)].dropna(subset=['year'])
    df_ent['year'] = df_ent['year'].astype(int)

    oa_sums = {
        'Diamond': 'diamond_sum',
        'Gold': 'gold_sum',
        'Green': 'green_sum',
        'Hybrid': 'hybrid_sum',
        'Bronze': 'bronze_sum',
        'Closed': 'closed_sum'
    }

    agg_dict = {col: 'sum' for col in oa_sums.values() if col in df_ent.columns}
    agg_dict['doc_count'] = 'sum'
    
    df_annual = df_ent.groupby('year').agg(agg_dict).reset_index()
    
    df_list = []
    for oa_type, sum_col in oa_sums.items():
        if sum_col in df_annual.columns:
            df_type = pd.DataFrame({
                'year': df_annual['year'],
                'Vía': oa_type,
                'Porcentaje': np.where(df_annual['doc_count'] > 0, (df_annual[sum_col] / df_annual['doc_count']) * 100, 0)
            })
            df_list.append(df_type)
            
    if not df_list:
        return None
        
    df_oa_melt = pd.concat(df_list, ignore_index=True)
    df_oa_melt = df_oa_melt.sort_values(['Vía', 'year'])

    fig = px.area(
        df_oa_melt,
        x="year",
        y="Porcentaje",
        color="Vía",
        color_discrete_map=OA_COLORS,
        labels={"Porcentaje": "Porcentaje (%)", "year": "Año"},
        title="Evolución Histórica de Vías de Acceso Abierto (%)",
        template="plotly_white",
        height=320
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(type='linear', tickformat='d'),
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )

    return fig

def render_geopolitical_quadrants(df_data, period_label):
    """
    Genera un scatter plot de cuadrantes cienciométricos (Bubble Chart) comparando 
    la producción y el impacto de las diferentes regiones del mundo.
    """
    if df_data is None or df_data.empty:
        return None

    # Filtrar solo registros de tipo 'Region' para comparar los bloques mundiales de forma justa
    df_reg = df_data[df_data['entity_type'] == 'Region'].copy()
    if df_reg.empty:
        return None

    # Filtrar por periodo
    if period_label == "Últimos 5 años (2021-2025)":
        df_reg = df_reg[(df_reg['year'] >= 2021) & (df_reg['year'] <= 2025)]
    
    # Agrupar por región para calcular promedios ponderados en el periodo
    df_calc = df_reg.copy()
    metrics_to_weight = ['fwci', 'percentile', 'pct_top_10', 'pct_top_1']
    for m in metrics_to_weight:
        if m in df_calc.columns:
            df_calc[f'{m}_prod'] = df_calc[m] * df_calc['doc_count']

    agg_dict = {'doc_count': 'sum'}
    for m in ['fwci', 'pct_top_10', 'pct_top_1']:
        if f'{m}_prod' in df_calc.columns:
            agg_dict[f'{m}_prod'] = 'sum'
            
    df_rank = df_calc.groupby('entity_name').agg(agg_dict).reset_index()

    for m in ['fwci', 'pct_top_10', 'pct_top_1']:
        if f'{m}_prod' in df_rank.columns:
            df_rank[m] = df_rank[f'{m}_prod'] / df_rank['doc_count']
            df_rank[m] = df_rank[m].fillna(0)

    # Limitar burbujas a valores positivos
    df_rank = df_rank[df_rank['doc_count'] > 0]

    if df_rank.empty:
        return None

    # Crear el gráfico de burbujas en cuadrantes
    fig = px.scatter(
        df_rank,
        x="doc_count",
        y="fwci",
        size="pct_top_10",
        color="entity_name",
        hover_name="entity_name",
        hover_data={"doc_count": True, "fwci": ":.2f", "pct_top_10": ":.1f%"},
        labels={
            "doc_count": "Volumen de Publicaciones (Artículos)",
            "fwci": "Impacto de Citación Normalizado (FWCI)",
            "pct_top_10": "% Top 10% Citados",
            "entity_name": "Región"
        },
        title=f"Posicionamiento Geopolítico por Región ({period_label})",
        template="plotly_white",
        height=450
    )

    # Añadir líneas de referencia para formar los 4 cuadrantes
    fig.add_hline(y=1.0, line_dash="dash", line_color="#ef4444", annotation_text="Media Mundial (FWCI = 1.0)", annotation_position="top left")
    
    # La línea vertical mediana de volumen para dividir producción
    median_docs = df_rank['doc_count'].median()
    fig.add_vline(x=median_docs, line_dash="dash", line_color="#94a3b8", annotation_text="Mediana de Volumen", annotation_position="top right")

    max_top10 = df_rank['pct_top_10'].max()
    sizeref = 2. * max_top10 / (40. ** 2) if max_top10 > 0 else 1.0
    fig.update_traces(marker=dict(sizemode='area', sizeref=sizeref, line=dict(width=1, color='DarkSlateGrey')))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    return fig

def render_sunburst_hierarchy(df_data, entity_name, selected_domain, selected_field, selected_subfield):
    """
    Genera un gráfico Sunburst (Sol Radiante) que muestra la composición jerárquica
    de la producción científica del subcampo desglosado por sus tópicos internos.
    """
    if df_data is None or df_data.empty:
        return None

    # Filtrar registros de la entidad seleccionada
    entity_type = 'Mundo' if entity_name == 'Mundo' else ('Country' if entity_name == 'México' else 'Region')
    lookup_name = 'Mundo' if entity_name == 'Mundo' else ('MX' if entity_name == 'México' else entity_name)
    
    df_ent = df_data[(df_data['entity_type'] == entity_type) & (df_data['entity_name'] == lookup_name)].copy()
    if df_ent.empty:
        return None

    # Agrupar por tópico para obtener la suma de producción
    df_topics = df_ent.groupby('topic')['doc_count'].sum().reset_index()
    df_topics = df_topics[df_topics['doc_count'] > 0]
    
    if df_topics.empty:
        return None

    # Agregar niveles de jerarquía
    df_topics['Domain'] = selected_domain
    df_topics['Field'] = selected_field
    df_topics['Subfield'] = selected_subfield
    
    # Limitar tópicos menores para evitar saturación visual (top 35)
    df_topics = df_topics.sort_values('doc_count', ascending=False).head(35)

    fig = px.sunburst(
        df_topics,
        path=['Domain', 'Field', 'Subfield', 'topic'],
        values='doc_count',
        color='doc_count',
        color_continuous_scale='Blues',
        title=f"Composición Jerárquica del Subcampo ({entity_name})",
        labels={"doc_count": "Artículos", "topic": "Tópico"},
        height=500
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def render_sdg_contributions(df_inst, entity_name, period_label):
    """
    Genera un gráfico de barras horizontales de las principales instituciones 
    líderes en contribución a los Objetivos de Desarrollo Sostenible (ODS).
    """
    if df_inst is None or df_inst.empty or 'sdg_docs' not in df_inst.columns:
        return None

    df_i = df_inst.copy()
    
    # Filtrar por entidad
    if entity_name == "Mundo":
        pass
    elif entity_name == "México":
        df_i = df_i[df_i['country_code'] == 'MX']
    else:
        df_i = df_i[df_i['region'] == entity_name]

    if df_i.empty:
        return None

    # Filtrar por periodo
    if period_label == "Últimos 5 años (2021-2025)":
        df_i = df_i[(df_i['year'] >= 2021) & (df_i['year'] <= 2025)]

    # Agrupar por institución sumando la producción total y los documentos ODS
    df_calc = df_i.groupby(['institution_name', 'country_code']).agg({
        'doc_count': 'sum',
        'sdg_docs': 'sum'
    }).reset_index()

    # Filtrar instituciones con alguna publicación ODS y ordenar
    df_calc = df_calc[df_calc['sdg_docs'] > 0]
    df_calc = df_calc.sort_values('sdg_docs', ascending=False).head(15)

    if df_calc.empty:
        return None

    # Calcular porcentaje de alineación
    df_calc['% Alineación ODS'] = (df_calc['sdg_docs'] / df_calc['doc_count'] * 100).round(1)
    df_calc['Institución'] = df_calc['institution_name'] + " (" + df_calc['country_code'] + ")"

    fig = px.bar(
        df_calc.sort_values('sdg_docs', ascending=True), # Barra horizontal pinta de abajo a arriba
        x="sdg_docs",
        y="Institución",
        orientation="h",
        color="% Alineación ODS",
        color_continuous_scale="Viridis",
        labels={"sdg_docs": "Documentos Alineados con ODS", "% Alineación ODS": "% de Alineación"},
        title=f"Líderes en Contribución al Desarrollo Sostenible (ODS) - {entity_name}",
        template="plotly_white",
        height=400
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig
