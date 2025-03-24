import pandas as pd
import numpy as np


def encode_date_index(df:pd.DataFrame, method='full-sin-cos'):
    '''
	Assumes that the index is a datetime.
    '''
    if method is None:
        return df
    df = df.sort_index()
    match method:
        case 'full-sin-cos':

            df["hour"] = df.index.hour
            df["day"] = df.index.day
            df["month"] = df.index.month
            df["days_in_month"] = df.index.days_in_month

            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            
            df["day_sin"] = np.sin(2 * np.pi * df["day"] / df["days_in_month"])  # Normalize by actual days
            df["day_cos"] = np.cos(2 * np.pi * df["day"] / df["days_in_month"])
            
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

            return df.drop(columns=["hour", "day", "month", "days_in_month"])
        
        case 'radial_months_days-sin-cos_hours':
            df["hour"] = df.index.hour
            df["day"] = df.index.day
            df["month"] = df.index.month
            df["days_in_month"] = df.index.days_in_month

            # Sin/Cos for hours
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            
            # Radial encoding for months
            angles_month = 2 * np.pi * (df["month"] - 1) / 12
            centers_month = np.arange(12) * 2 * np.pi / 12
            sigma_month = np.pi / 12
            diff_month = np.abs(angles_month.values[:, None] - centers_month[None, :])
            diff_month = np.minimum(diff_month, 2 * np.pi - diff_month)
            rbf_months = np.exp(- (diff_month ** 2) / (2 * sigma_month ** 2))
            for i in range(12):
                df[f"month_rbf_{i+1}"] = rbf_months[:, i]

            # Radial encoding for days
            angles_day = 2 * np.pi * (df["day"] - 1) / df["days_in_month"]
            centers_day = np.linspace(0, 2 * np.pi, df["days_in_month"].max(), endpoint=False)
            sigma_day = np.pi / df["days_in_month"].max()
            diff_day = np.abs(angles_day.values[:, None] - centers_day[None, :])
            diff_day = np.minimum(diff_day, 2 * np.pi - diff_day)
            rbf_days = np.exp(- (diff_day ** 2) / (2 * sigma_day ** 2))
            for i in range(df["days_in_month"].max()):
                df[f"day_rbf_{i+1}"] = rbf_days[:, i]

            return df.drop(columns=["hour", "day", "month", "days_in_month"])
        
        case 'radial_months-sin-cos_days_hours':
            df["hour"] = df.index.hour
            df["day"] = df.index.day
            df["month"] = df.index.month
            df["days_in_month"] = df.index.days_in_month

            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            
            df["day_sin"] = np.sin(2 * np.pi * df["day"] / df["days_in_month"])
            df["day_cos"] = np.cos(2 * np.pi * df["day"] / df["days_in_month"])
            
            # Radial encoding for months:
            # Convert month to an angle (0 to 2π) with month 1 mapped to 0
            angles = 2 * np.pi * (df["month"] - 1) / 12
            # Define centers for 12 months (each center in radians)
            centers = np.arange(12) * 2 * np.pi / 12
            # Choose a bandwidth; here σ is set to pi/12 (adjustable)
            sigma = np.pi / 12
            # Compute the absolute difference between each angle and each center
            diff = np.abs(angles.values[:, None] - centers[None, :])
            # Because the month is periodic, take the minimum of the direct difference and the wrap-around difference
            diff = np.minimum(diff, 2 * np.pi - diff)
            # Compute the radial basis (Gaussian) values
            rbf_features = np.exp(- (diff ** 2) / (2 * sigma ** 2))
            # Create new columns for each radial basis feature (one for each month center)
            for i in range(12):
                df[f"month_rbf_{i+1}"] = rbf_features[:, i]
            
            return df.drop(columns=["hour", "day", "month", "days_in_month"])
        
        case _:
            print('Unsupported method')
            return None

# ... soon 