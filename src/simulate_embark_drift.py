import pandas as pd

# Load the original data
df = pd.read_csv('data/titanic.csv')

# Introduce drift: Change the Embarked port for all Pclass 3 passengers from 'S' to 'C'
# Use .loc to avoid a SettingWithCopyWarning
df.loc[(df['Pclass'] == 3) & (df['Embarked'] == 'S'), 'Embarked'] = 'C'

# Save the new, drifted data
df.to_csv('data/titanic_drifted.csv', index=False)

print("Embarked port for Pclass 3 passengers from 'S' changed to 'C'. New data saved to data/titanic_drifted.csv")
