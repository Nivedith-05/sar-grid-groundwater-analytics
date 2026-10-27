import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def simulate_sar_soil_moisture():
    print("=== 1. SATELLITE SAR RADAR SOIL MOISTURE ANALYTICS ===")
    data = {
        'field_id': [f'FIELD_{i:03d}' for i in range(1, 6)],
        'latitude': [12.8342, 12.8350, 12.8311, 12.8390, 12.8325],
        'longitude': [79.7031, 79.7045, 79.6990, 79.7088, 79.7012],
        'soil_moisture_pct': [42.5, 18.2, 68.0, 14.5, 50.1]
    }
    df = pd.DataFrame(data)
    conditions = [
        (df['soil_moisture_pct'] < 20.0),
        (df['soil_moisture_pct'] >= 20.0) & (df['soil_moisture_pct'] <= 55.0),
        (df['soil_moisture_pct'] > 55.0)
    ]
    choices = ['CRITICAL_DRY (Trigger Alert)', 'OPTIMAL', 'OVER_IRRIGATED (Cutoff Alert)']
    df['hydration_status'] = np.select(conditions, choices, default='UNKNOWN')
    print(df.to_string(index=False))
    print("\n")
    return df

def simulate_grid_dry_run_detection():
    print("=== 2. DISCOM SUBSTATION POWER ANALYTICS (ISOLATION FOREST) ===")
    telemetry_data = np.array([
        [24.5, 0.85], [22.1, 0.83], [23.8, 0.88],
        [8.2,  0.51], [7.5,  0.48], [21.0, 0.82]
    ])
    model = IsolationForest(contamination=0.3, random_state=42)
    model.fit(telemetry_data)
    predictions = model.predict(telemetry_data)
    
    results = []
    for idx, (current, pf) in enumerate(telemetry_data):
        is_anomaly = predictions[idx] == -1
        status = "ALERT: PUMP CAVITATION / DRY-RUN DETECTED!" if is_anomaly else "NORMAL"
        results.append({
            'transformer_id': f'XMER_AG_{idx+1:02d}',
            'current_amps': current,
            'power_factor': pf,
            'grid_status': status
        })
    df_grid = pd.DataFrame(results)
    print(df_grid.to_string(index=False))
    print("\n")
    return df_grid

if __name__ == "__main__":
    print("Starting Zero-Hardware Groundwater & Grid Analytics Prototype...\n")
    simulate_sar_soil_moisture()
    simulate_grid_dry_run_detection()
    print("Execution Completed Successfully.")
