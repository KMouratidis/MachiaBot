import sqlite3
from data_processing import process_data

# TODO: Add nltk corpus to the DB for quicker lookup (?)
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

# TODO: add keyword argument about the speaker when I get more data
def fetch_all(db="MachiaBot.db"):
    conn = sqlite3.connect(db)
    curr = conn.cursor()

    curr.execute("SELECT * FROM dataset WHERE speaker='Machiavelli'")
    machiavelli_ = curr.fetchall()

    curr.execute("SELECT * FROM dataset WHERE speaker='Montesquieu'")
    montesquieu_ = curr.fetchall()

    curr.close()
    conn.close()

    # No need to have labels of who speaks since we already split the data
    machiavelli_, montesquieu_ = [x[1] for x in machiavelli_], [x[1] for x in montesquieu_]

    return machiavelli_, montesquieu_

if __name__ == "__main__":
    create_sqlite_db()