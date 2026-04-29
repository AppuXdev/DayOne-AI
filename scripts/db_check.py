from sqlalchemy import create_engine, text
import os, traceback
url = os.getenv('DATABASE_URL', 'postgresql+psycopg://dayone:dayone@localhost:55432/dayone')
print('DATABASE_URL=' + url)
try:
    engine = create_engine(url)
    with engine.connect() as conn:
        r = conn.execute(text('SELECT usename FROM pg_user;'))
        rows = r.fetchall()
        print('pg_user rows =', rows)
except Exception as e:
    print('ERROR:', type(e).__name__, str(e))
    traceback.print_exc()
