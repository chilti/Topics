import os
from dotenv import load_dotenv
load_dotenv()

print("CH_HOST:", os.environ.get('CH_HOST'))
print("CH_PORT:", os.environ.get('CH_PORT'))
print("CH_USER:", os.environ.get('CH_USER'))
# Imprimir longitud de contraseña para verificar interpolación de $
pwd = os.environ.get('CH_PASSWORD', '')
print("CH_PASSWORD length:", len(pwd))
print("CH_PASSWORD starts with $:", pwd.startswith('$'))
print("CH_PASSWORD value (safe check):", pwd[:3] + "..." + pwd[-3:] if len(pwd) > 6 else pwd)

import clickhouse_connect
try:
    client = clickhouse_connect.get_client(
        host=os.environ.get('CH_HOST'),
        port=int(os.environ.get('CH_PORT', 8123)),
        username=os.environ.get('CH_USER'),
        password=pwd,
        database=os.environ.get('CH_DATABASE', 'rag'),
        secure=True,
        verify=False,
        connect_timeout=30,
        send_receive_timeout=300
    )
    print("Conexión con SSL exitosa!")
except Exception as e:
    print("Conexión con SSL falló con error:", type(e), e)
    try:
        client = clickhouse_connect.get_client(
            host=os.environ.get('CH_HOST'),
            port=int(os.environ.get('CH_PORT', 8123)),
            username=os.environ.get('CH_USER'),
            password=pwd,
            database=os.environ.get('CH_DATABASE', 'rag'),
            secure=False,
            verify=False,
            connect_timeout=30,
            send_receive_timeout=300
        )
        print("Conexión SIN SSL exitosa!")
    except Exception as e2:
        print("Conexión SIN SSL también falló con error:", type(e2), e2)

