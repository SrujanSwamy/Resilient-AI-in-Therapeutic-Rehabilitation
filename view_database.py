import sqlite3

db = sqlite3.connect('demo_results.db')
cursor = db.cursor()

def show_table(table_name):
    print(f"\n{'='*60}")
    print(f"TABLE: {table_name}")
    print('='*60)
    
    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]
    print("COLUMNS:", columns)
    print('-'*60)
    
    # Get data
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    if not rows:
        print("(No data)")
    else:
        for row in rows:
            for col, val in zip(columns, row):
                print(f"  {col}: {val}")
            print()

show_table('users')
show_table('sessions')
show_table('repetitions')
show_table('performance_metrics')

db.close()