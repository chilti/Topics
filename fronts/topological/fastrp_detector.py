from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración Neo4j (Reutilizando parámetros de RAGs/Mexico)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD_MEXICO", "password123")

class TopologicalFrontsManager:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def project_sandbox_graph(self, graph_name="sandbox_graph"):
        """
        Crea una proyección en memoria de GDS para el sandbox.
        """
        with self.driver.session() as session:
            # Primero eliminar si existe
            session.run(f"CALL gds.graph.drop('{graph_name}', false)")
            
            # Proyectar Work nodes y sus relaciones CITATION
            query = f"""
            CALL gds.graph.project(
              '{graph_name}',
              'Work',
              {{
                CITATION: {{
                  type: 'CITATION',
                  orientation: 'UNDIRECTED'
                }}
              }}
            )
            """
            session.run(query)
            print(f"   [Topological] Grafo '{graph_name}' proyectado en GDS.")

    def run_fastrp(self, graph_name="sandbox_graph", embedding_dim=128):
        """
        Ejecuta FastRP y retorna los embeddings.
        """
        query = f"""
        CALL gds.fastRP.stream(
          '{graph_name}',
          {{
            embeddingDimension: {embedding_dim},
            randomSeed: 42
          }}
        )
        YIELD nodeId, embedding
        RETURN gds.util.asNode(nodeId).id AS work_id, embedding
        """
        with self.driver.session() as session:
            result = session.run(query)
            return {record["work_id"]: record["embedding"] for record in result}

def ingest_sandbox_to_neo4j(df, driver_uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS):
    """
    Ingesta el subset del sandbox en Neo4j para poder correr FastRP.
    """
    driver = GraphDatabase.driver(driver_uri, auth=(user, password))
    
    # Preparar datos
    # Solo necesitamos IDs y relaciones de cita que estén dentro del set
    work_ids = set(df['id'].tolist())
    
    rels = []
    for _, row in df.iterrows():
        source = row['id']
        for target in row['referenced_works']:
            target_clean = target.strip('"')
            if target_clean in work_ids:
                rels.append({"source": source, "target": target_clean})
                
    with driver.session() as session:
        print("   [Topological] Limpiando sandbox previo en Neo4j...")
        session.run("MATCH (n:Work) DETACH DELETE n")
        
        print(f"   [Topological] Cargando {len(work_ids)} nodos y {len(rels)} relaciones...")
        # Carga masiva de nodos
        session.run("""
        UNWIND $ids AS work_id
        CREATE (:Work {id: work_id})
        """, ids=list(work_ids))
        
        # Carga masiva de relaciones
        session.run("""
        UNWIND $rels AS rel
        MATCH (s:Work {id: rel.source}), (t:Work {id: rel.target})
        CREATE (s)-[:CITATION]->(t)
        """, rels=rels)
        
    driver.close()
    print("   [Topological] Ingesta finalizada.")
