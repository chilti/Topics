"""
Módulo de Definición de Regiones Mundiales
Agrupa los países (vía códigos ISO-2) en 8 grandes bloques representativos 
del Norte y Sur Global, junto con una categoría dedicada a China.
"""

GLOBAL_REGIONS = {
    # --- POTENCIA GLOBAL ---
    'China': ['CN', 'HK'],
    
    # --- GLOBAL SOUTH ---
    'Asia Emergente': ['IN', 'ID', 'MY', 'PH', 'TH', 'VN', 'PK', 'BD'],
    'Latinoamérica y Caribe': ['AR', 'BO', 'BR', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'SV', 'GT', 'HN', 'MX', 'NI', 'PA', 'PY', 'PE', 'PR', 'UY', 'VE', 'JM', 'TT'],
    'África Subsahariana': ['ZA', 'NG', 'KE', 'ET', 'GH', 'TZ', 'UG', 'SN', 'CM', 'CI', 'ZW'],
    'MENA': ['DZ', 'EG', 'IR', 'IQ', 'JO', 'KW', 'LB', 'LY', 'MA', 'OM', 'PS', 'QA', 'SA', 'SY', 'TN', 'AE', 'YE'],
    
    # --- GLOBAL NORTH ---
    'Norteamérica Anglosajona': ['US', 'CA', 'GB', 'IE', 'AU', 'NZ'],
    'Europa Central/Occidental': ['AT', 'BE', 'CH', 'DE', 'DK', 'ES', 'FI', 'FR', 'GR', 'IT', 'LU', 'NL', 'NO', 'PT', 'SE', 'IS', 'IL'],
    'Europa del Este': ['PL', 'CZ', 'HU', 'RO', 'BG', 'SK', 'HR', 'EE', 'LV', 'LT', 'UA', 'RU'],
    'Asia-Pacífico Desarrollado': ['JP', 'KR', 'SG', 'TW']
}

def get_all_country_codes():
    """Devuelve la lista plana de todos los códigos ISO de país monitoreados."""
    return [c for countries in GLOBAL_REGIONS.values() for c in countries]

def get_region_for_country(country_code):
    """Devuelve el nombre de la región a la que pertenece un país dado."""
    for region, countries in GLOBAL_REGIONS.items():
        if country_code in countries:
            return region
    return 'Other'
