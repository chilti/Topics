from Visualizer import NetworkEngine
import networkx as nx

def main():
    # 1. Instanciar el motor
    net = NetworkEngine()
    
    print("🚀 Generando red de ejemplo multicapa...")

    # 2. Agregar Nodos de diferentes tipos (metadatos tipo CoAuthra)
    # Autores (Nodos principales)
    net.add_node("A1", label="Dr. Arriaga", node_type="author", weight=15)
    net.add_node("A2", label="Dra. Benítez", node_type="author", weight=10)
    net.add_node("A3", label="Dr. Castillo", node_type="author", weight=8)
    
    # Instituciones
    net.add_node("I1", label="UNAM", node_type="institution", weight=25)
    net.add_node("I2", label="IPN", node_type="institution", weight=20)
    
    # Países
    net.add_node("C1", label="México", node_type="country", weight=30)
    net.add_node("C2", label="España", node_type="country", weight=25)
    
    # Trabajos (Papers)
    net.add_node("W1", label="Deep Learning in Bio", node_type="paper", weight=5)
    net.add_node("W2", label="Climate Change Study", node_type="paper", weight=5)

    # 3. Crear conexiones (Edges)
    # Autores colaborando
    net.add_edge("A1", "A2", weight=5)
    net.add_edge("A1", "A3", weight=2)
    
    # Autores en Instituciones
    net.add_edge("A1", "I1", weight=10)
    net.add_edge("A2", "I1", weight=10)
    net.add_edge("A3", "I2", weight=10)
    
    # Instituciones en Países
    net.add_edge("I1", "C1", weight=15)
    net.add_edge("I2", "C1", weight=15)
    
    # Autores publicando Papers
    net.add_edge("A1", "W1", weight=1)
    net.add_edge("A2", "W1", weight=1)
    net.add_edge("A3", "W2", weight=1)

    # 4. Calcular Comunidades Automáticamente (Leiden Algorithm)
    print("🔍 Calculando clusters de investigación...")
    net.compute_communities()

    # 5. Exportar a HTML (Estilo CoAuthra / D3.js)
    html_file = "Network_CoAuthra_Style.html"
    net.export_d3(html_file, title="Mapa de Colaboración - Proyecto X")
    
    # 6. Exportar a VOSviewer (JSON)
    vos_file = "vos_map.json"
    net.export_vosviewer(vos_file)

    print("-" * 40)
    print(f"✨ ¡Éxito! Archivos generados:")
    print(f"   1. Visualización D3: {html_file}")
    print(f"   2. Mapa VOSviewer: {vos_file}")
    print("-" * 40)
    print("Sugerencia: Abre el archivo HTML en tu navegador para ver la interactividad.")

if __name__ == "__main__":
    main()
