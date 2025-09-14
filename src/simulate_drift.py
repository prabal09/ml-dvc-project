# simulate_drift.py
import pandas as pd
df = pd.read_csv('data/titanic.csv')
df['Age'] = df['Age'] * 1.05  # Increase all ages by 5%
df.to_csv('data/train_drifted.csv', index=False)
