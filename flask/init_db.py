import sqlite3

connection = sqlite3.connect('database.db')
sample = "Art. L121-1\nLa commercialisation des établissements publics de coopération intercommunale en application de l'article L. 113-3 est publiée au Journal officiel de l'Union européenne et à l'accomplissement de la police nationale d'affaires par le greffe du tribunal.\nArt. L121-5\nLe président du conseil territorial est applicable en Nouvelle-Calédonie, sous réserve des adaptations prévues par l'article L. 124-9, des articles L. 123-14 et suivants :\n1° Le conseil régional de santé et des propriétaire"

with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO gen_samples (content) VALUES (?)",
            (sample,)
            )

connection.commit()
connection.close()