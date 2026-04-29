from sqlalchemy import create_engine, text
import traceback
url = "postgresql+psycopg://dayone:dayone@172.24.0.2:5432/dayone"
print('CONNECT_URL=', url)
try:
    engine = create_engine(url)
    with engine.connect() as conn:
        print('SELECT 1 ->', conn.execute(text('SELECT 1;')).scalar())
except Exception as e:
    print('ERROR:', type(e).__name__, str(e))
    traceback.print_exc()
