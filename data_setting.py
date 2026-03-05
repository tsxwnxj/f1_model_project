import pandas as pd
import numpy as np


# 1. 데이터 로드
df = pd.read_csv("f1_2018_2024_laps_dataset.csv")
print("Original shape:", df.shape)

# 2. PitStop 여부 생성
df["PitStop"] = df["PitInTime"].notna().astype(int)

# 3. Pit time -> seconds
def pitTime_to_seconds(pittime):
    if pd.isna(pittime):
        return np.nan
    
    minutes, seconds = pittime.split(":")
    return float(minutes) * 60 + float(seconds)

df["PitIn_seconds"] = pd.to_timedelta(df["PitInTime"]).dt.total_seconds()
df["PitOut_seconds"] = pd.to_timedelta(df["PitOutTime"]).dt.total_seconds()

df["PitDuration"] = df["PitOut_seconds"] - df["PitIn_seconds"]

# 4. 데이터 정렬
df = df.sort_values(
    ["Season", "Round", "Driver", "LapNumber"]
).reset_index(drop=True)

# 5. Position 변화량 계산
df["PositionChange"] = (
    df.groupby(["Season", "Round", "Driver"])["Position"]
    .diff()
)

df["PositionChange"] = df["PositionChange"].fillna(0)

# 6. gap to leader 계산
df["LeaderLapTime"] = (
    df.groupby(["Season", "Round", "LapNumber"])["LapTime"]
    .transform("min")
)

df["GapToLeader"] = df["LapTime"] - df["LeaderLapTime"]

# 7. gap to car ahead 계산
df = df.sort_values(
    ["Season", "Round", "LapNumber", "Position"]
)

df["AheadLapTime"] = (
    df.groupby(["Season", "Round", "LapNumber"])["LapTime"]
    .shift(1)
)

df["GapToAhead"] = df["LapTime"] - df ["AheadLapTime"]
df["GapToAhead"] = df["GapToAhead"].fillna(0)

# 8. Lap Progress
df["MaxLap"] = (
    df.groupby(["Season", "Round"])["LapNumber"]
    .transform("max")
)

df["LapProgress"] = df["LapNumber"] / df["MaxLap"]

# 9. Stint progress
df["StintMax"] = (
    df.groupby(["Season", "Round", "Driver", "Stint"])["TyreLife"]
    .transform("max")
)

df["StintProgress"] = df["TyreLife"] / df["StintMax"]

# 10. 평균 팀 페이스
df["TeamAvgPace"] = (
    df.groupby(["Season", "Round", "Team"])["LapTime"]
    .transform("mean")
)

# 11. 평균 드라이버 페이스
df["DriverAvgPace"] = (
    df.groupby(["Season", "Round", "Driver"])["LapTime"]
    .transform("mean")
)

# Categorical encoding
categorical_cols = [
    "Driver",
    "Team",
    "Compound",
    "Race"
]

df = pd.get_dummies(
    df,
    columns = categorical_cols,
    drop_first = True
)

# TrackStatus encoding
df["TrackStatus"] = df["TrackStatus"].astype("category")

df = pd.get_dummies(
    df,
    columns = ["TrackStatus"],
    drop_first = True
)

# 불필요 컬럼 제거
drop_cols = [
    "PitInTime",
    "PitOutTime",
    "LeaderLapTime",
    "AheadLapTime"
]

df = df.drop(columns = drop_cols)

# 결측치 처리
df["PitDuration"] = df["PitDuration"].fillna(0)

df = df.dropna(subset = ["LapTime"])

# 최종 데이터 확인
print("Processed shape: ", df.shape)
print(df.head())

# 저장
df.to_csv(
    "f1_processed_dataset.csv",
    index=False
)

print("Saved: f1_processed_dataset(2018-2024).csv")