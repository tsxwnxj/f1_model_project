import fastf1
import pandas as pd
from tqdm import tqdm

# 캐시 설정 (속도 매우 중요)
fastf1.Cache.enable_cache('f1_cache')

seasons = list(range(2025, 2026))

dataset = []

for year in seasons:

    schedule = fastf1.get_event_schedule(year)

    for _, event in tqdm(schedule.iterrows(), desc=f"{year} season"):

        try:
            session = fastf1.get_session(year, event['RoundNumber'], 'R')
            session.load()

            laps = session.laps

            df = laps[[
                'Driver',
                'Team',
                'LapNumber',
                'LapTime',
                'Stint',
                'Compound',
                'TyreLife',
                'Position',
                'TrackStatus',
                'PitInTime',
                'PitOutTime'
            ]].copy()

            df['LapTime'] = df['LapTime'].dt.total_seconds()

            df['Season'] = year
            df['Race'] = event['EventName']
            df['Round'] = event['RoundNumber']

            dataset.append(df)

        except Exception as e:
            print(f"Error loading {year} round {event['RoundNumber']}")

dataset = pd.concat(dataset)

dataset = dataset.dropna(subset=['LapTime'])

dataset.to_csv("f1_2025_laps_dataset.csv", index=False)

print(dataset.head())