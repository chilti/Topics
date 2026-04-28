
from dotenv import load_dotenv
load_dotenv()
from pipeline_topic.compute_metrics_flat import get_ch_client
client = get_ch_client()
print('Con duplicates:')
print(client.query_df('SELECT count() FROM works_flat WHERE subfield_name = \'Pulmonary and Respiratory Medicine\' AND publication_year BETWEEN 2021 AND 2025').iloc[0,0])
print('Sin duplicates LIMIT 1:')
print(client.query_df('SELECT count() FROM (SELECT * FROM works_flat WHERE subfield_name = \'Pulmonary and Respiratory Medicine\' AND publication_year BETWEEN 2021 AND 2025 LIMIT 1 BY id)').iloc[0,0])

