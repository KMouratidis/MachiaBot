import sqlite3
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize


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


machiavelli, montesquieu = fetch_all()

machia_tokens = [word_tokenize(sent) for sent in machiavelli]

# Lemmatize words
lemmatizer = WordNetLemmatizer()
machia_lemmas = [[lemmatizer.lemmatize(word) for word in sent] for sent in machia_tokens]

for tokens in machia_lemmas[:5]:
    print(tokens)

# TODO: tokenize words by creating a voting model (maybe weighted) aggregating various tokenizers