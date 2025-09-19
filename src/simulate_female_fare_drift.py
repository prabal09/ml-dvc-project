import pandas as pd

# Load the original data
df = pd.read_csv('data/titanic.csv')

# Introduce drift: Increase the fare for all female passengers by 50%
df.loc[df['Sex'] == 'female', 'Fare'] = df.loc[df['Sex'] == 'female', 'Fare'] * 1.5

# Save the new, drifted data
df.to_csv('data/titanic_drifted.csv', index=False)

print("Fare for female passengers increased by 50%. New data saved to data/titanic_drifted.csv")
