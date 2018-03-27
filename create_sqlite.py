import sqlite3
from data_processing import process_data

def create_sqlite_db():
    data = process_data()

    conn = sqlite3.connect("MachiaBot.db")
    curr = conn.cursor()

    curr.execute("CREATE TABLE IF NOT EXISTS dataset (speaker TEXT, dialogue TEXT)")

    for speaker, dialogue in data:
        curr.execute("INSERT INTO dataset VALUES (?, ?)", (speaker, dialogue))

    conn.commit()

    curr.close()
    conn.close()

if __name__ == "__main__":
    create_sqlite_db()