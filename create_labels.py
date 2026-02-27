import pandas as pd

df = pd.read_csv("health_data.csv")

def calculate_risk(row):
    if row['heart_rate'] > 100 and row['spo2'] < 95:
        return 1
    elif row['temperature'] > 37.5:
        return 1
    else:
        return 0

df['risk'] = df.apply(calculate_risk, axis=1)

df.to_csv("health_data_labeled.csv", index=False)

print("Dataset with labels created")