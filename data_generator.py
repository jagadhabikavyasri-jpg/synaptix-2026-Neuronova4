import pandas as pd
import numpy as np
import time
import os

file_name = "health_data.csv"

def generate_health_data():
    data = {
        "heart_rate": np.random.randint(60,110),
        "spo2": np.random.randint(92,100),
        "temperature": round(np.random.uniform(36,38),2),
        "activity": np.random.randint(0,10),
        "hrv": np.random.randint(20,100)
    }
    return data

while True:
    health_data = generate_health_data()

    df = pd.DataFrame([health_data])

    # If file doesn't exist create it
    if not os.path.isfile(file_name):
        df.to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, mode='a', header=False, index=False)

    print(health_data)

    time.sleep(1)