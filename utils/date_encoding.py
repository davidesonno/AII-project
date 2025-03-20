import pandas as pd
import numpy as np


def encode_date(df:pd.DataFrame, column='Date', method='full-sin-cos'):
    df[column] = pd.to_datetime(df[column])
    df = df.sort_values(by=column)
    match method:
        case 'full-sin-cos':

            df["hour"] = df[column].dt.hour
            df["day"] = df[column].dt.day
            df["month"] = df[column].dt.month
            df["days_in_month"] = df[column].dt.days_in_month

            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            
            df["day_sin"] = np.sin(2 * np.pi * df["day"] / df["days_in_month"])  # Normalize by actual days
            df["day_cos"] = np.cos(2 * np.pi * df["day"] / df["days_in_month"])
            
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

            return df.drop(columns=[column, "hour", "day", "month", "days_in_month"])
            
        case _:
            print('Unsupported method')
            return None

# ... soon 