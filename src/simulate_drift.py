# simulate_drift.py
import pandas as pd
df = pd.read_csv('data/titanic.csv')
df['Age'] = df['Age'] * 1.15  # Increase all ages by 15%
df.to_csv('data/titanic_drifted.csv', index=False)
