from ase.db import connect

db = connect('organometal.db')
row = next(db.select(project='organometal'))
print(f'Name of the first row: {row.name}')
number_of_compounds = db.count(project='organometal')
print(f'Number of rows: {number_of_compounds}')
