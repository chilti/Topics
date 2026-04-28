import networkx as nx
import json
import os
import igraph as ig
import leidenalg
import pandas as pd

class NetworkEngine:
    """
    Motor de Visualización de Redes Científicas.
    Permite exportar grafos de NetworkX a formatos interactivos de alta calidad:
    1. D3.js (Inspirado en la estética de CoAuthra)
    2. VOSviewer (Estándar de bibliometría para mapas de densidad)
    """

    def __init__(self, G=None):
        self.G = G if G is not None else nx.Graph()

    def add_node(self, node_id, label=None, node_type='paper', weight=1, **kwargs):
        """Agrega un nodo con metadatos."""
        self.G.add_node(node_id, label=label or str(node_id), type=node_type, weight=weight, **kwargs)

    def add_edge(self, u, v, weight=1, **kwargs):
        """Agrega una arista con peso."""
        self.G.add_edge(u, v, weight=weight, **kwargs)

    def compute_communities(self):
        """Calcula clusters usando el algoritmo de Leiden (vía igraph)."""
        if self.G.number_of_nodes() == 0: return
        
        # Convertir NetworkX a igraph para usar leidenalg
        g_ig = ig.Graph.from_networkx(self.G)
        # Asegurar que el grafo tenga pesos para que el clustering sea preciso
        weights = [e.get('weight', 1.0) for e in self.G.edges.values()]
        
        partition = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition, weights=weights)
        
        # Re-asignar clusters a NetworkX
        clusters = {node: partition.membership[i] for i, node in enumerate(self.G.nodes())}
        nx.set_node_attributes(self.G, clusters, 'cluster')
        return clusters

    def export_vosviewer(self, output_path):
        """Genera un archivo JSON para VOSviewer Online (https://app.vosviewer.com/)."""
        items = []
        for node, data in self.G.nodes(data=True):
            items.append({
                "id": str(node),
                "label": data.get('label', str(node)),
                "cluster": data.get('cluster', 0),
                "weight": data.get('weight', 1),
                "type": data.get('type', 'node')
            })
        
        links = []
        for u, v, data in self.G.edges(data=True):
            links.append({
                "source_id": str(u),
                "target_id": str(v),
                "strength": data.get('weight', 1)
            })
            
        vos_data = {
            "network": {
                "items": items,
                "links": links
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vos_data, f, indent=2, ensure_ascii=False)
        return output_path

    def get_d3_html(self, title="Network Explorer"):
        """Retorna el string HTML con la visualización D3.js."""
        nodes = []
        for n, d in self.G.nodes(data=True):
            nodes.append({
                "id": str(n),
                "label": d.get('label', str(n)),
                "type": d.get('type', 'node'),
                "weight": d.get('weight', 1),
                "cluster": d.get('cluster', 0)
            })
        
        links = []
        for u, v, d in self.G.edges(data=True):
            links.append({
                "source": str(u),
                "target": str(v),
                "weight": d.get('weight', 1)
            })

        data_json = json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0d0f14;
            --surface: #1a1d24;
            --accent: #4fc3f7;
            --text: #dde1ec;
            --muted: #565c78;
            --border: #2a2d38;
        }}
        body, html {{ margin:0; padding:0; width:100%; height:100%; background: var(--bg); font-family: 'IBM Plex Sans', sans-serif; color: var(--text); overflow: hidden; }}
        #controls {{ position: fixed; top: 10px; left: 10px; z-index: 10; background: var(--surface); padding: 10px; border-radius: 6px; border: 1px solid var(--border); box-shadow: 0 5px 15px rgba(0,0,0,0.5); width: 220px; pointer-events: auto; }}
        h1 {{ font-size: 12px; margin: 0 0 8px 0; color: var(--accent); text-transform: uppercase; letter-spacing: 1px; }}
        .search-box {{ width: 100%; background: #000; border: 1px solid var(--border); color: #fff; padding: 5px; border-radius: 3px; outline: none; margin-bottom: 10px; font-size: 11px; }}
        .search-box:focus {{ border-color: var(--accent); }}
        .stats {{ font-size: 10px; color: var(--muted); display: flex; gap: 10px; }}
        #canvas {{ width: 100%; height: 100%; cursor: move; }}
        .node {{ cursor: pointer; stroke: var(--bg); stroke-width: 1.5px; transition: stroke 0.2s; }}
        .node:hover {{ stroke: #fff; stroke-width: 2px; }}
        .link {{ stroke: #444; stroke-opacity: 0.4; }}
        .label {{ font-size: 10px; fill: var(--text); pointer-events: none; opacity: 0.8; font-weight: 300; }}
        .tooltip {{ position: fixed; background: var(--surface); border: 1px solid var(--accent); padding: 8px 12px; border-radius: 4px; font-size: 12px; pointer-events: none; opacity: 0; transition: opacity 0.2s; z-index: 100; }}
    </style>
</head>
<body>
    <div id="controls">
        <h1>{title}</h1>
        <input type="text" class="search-box" id="search" placeholder="Buscar..." oninput="filterNodes()">
        <div class="stats"><span id="node-count">0 nodes</span><span id="link-count">0 links</span></div>
    </div>
    <div class="tooltip" id="tooltip"></div>
    <svg id="canvas"></svg>
    <script>
        const data = {data_json};
        
        // Sizing robusto para iframes (evita width/height 0)
        let width = window.innerWidth || 800;
        let height = window.innerHeight || 600;
        
        const svg = d3.select("#canvas")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("viewBox", [0, 0, width, height]);

        if (data.nodes.length === 0) {
            d3.select("body").append("div")
                .style("position", "absolute")
                .style("top", "50%")
                .style("left", "50%")
                .style("transform", "translate(-50%, -50%)")
                .style("color", "#888")
                .text("No hay datos para mostrar en la red.");
        }
        const g = svg.append("g");
        svg.call(d3.zoom().scaleExtent([0.1, 8]).on("zoom", (e) => g.attr("transform", e.transform)));
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(60))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.weight) * 3 + 10));
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        const link = g.append("g").selectAll("line").data(data.links).join("line").attr("class", "link").attr("stroke-width", d => Math.sqrt(d.weight) + 0.5);
        const node = g.append("g").selectAll("circle").data(data.nodes).join("circle").attr("class", "node")
            .attr("r", d => Math.sqrt(d.weight) * 3 + 5).attr("fill", d => colorScale(d.cluster))
            .on("mouseover", showTooltip).on("mouseout", hideTooltip)
            .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended));
        const labels = g.append("g").selectAll("text").data(data.nodes).join("text").attr("class", "label").attr("dx", 10).attr("dy", 4).text(d => d.label);
        simulation.on("tick", () => {{
            link.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
            node.attr("cx", d => d.x).attr("cy", d => d.y);
            labels.attr("x", d => d.x).attr("y", d => d.y);
        }});
        function showTooltip(event, d) {{
            const tt = d3.select("#tooltip");
            tt.html(`<strong>${{d.label}}</strong><br>Type: ${{d.type}}<br>Cluster: ${{d.cluster}}`)
              .style("left", (event.pageX + 10) + "px").style("top", (event.pageY - 10) + "px").style("opacity", 1);
        }}
        function hideTooltip() {{ d3.select("#tooltip").style("opacity", 0); }}
        function filterNodes() {{
            const val = document.getElementById("search").value.toLowerCase();
            node.style("opacity", d => d.label.toLowerCase().includes(val) ? 1 : 0.1);
            labels.style("opacity", d => d.label.toLowerCase().includes(val) ? 1 : 0.1);
            link.style("opacity", d => (d.source.label.toLowerCase().includes(val) || d.target.label.toLowerCase().includes(val)) ? 0.4 : 0.05);
        }}
        function dragstarted(e) {{ if (!e.active) simulation.alphaTarget(0.3).restart(); e.subject.fx = e.subject.x; e.subject.fy = e.subject.y; }}
        function dragged(e) {{ e.subject.fx = e.x; e.subject.fy = e.y; }}
        function dragended(e) {{ if (!e.active) simulation.alphaTarget(0); e.subject.fx = null; e.subject.fy = null; }}
        document.getElementById("node-count").innerText = `${{data.nodes.length}} nodes`;
        document.getElementById("link-count").innerText = `${{data.links.length}} links`;
    </script>
</body>
</html>"""

    def export_d3(self, output_path, title="Network Explorer"):
        """Genera un archivo HTML con la visualización."""
        html = self.get_d3_html(title)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return output_path

if __name__ == "__main__":
    # Test rápido
    engine = NetworkEngine()
    engine.add_node("A", label="Albert Einstein", weight=10)
    engine.add_node("B", label="Marie Curie", weight=12)
    engine.add_node("C", label="Isaac Newton", weight=8)
    engine.add_edge("A", "B", weight=5)
    engine.add_edge("B", "C", weight=2)
    
    engine.compute_communities()
    engine.export_d3("test_network.html")
    print("Demo finalizado.")
