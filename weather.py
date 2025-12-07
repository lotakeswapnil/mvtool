# weather_api.py
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from typing import Tuple, Dict

def make_openmeteo_client(cache_name: str = ".cache", expire_after: int = -1,
                          retries: int = 5, backoff: float = 0.2) -> openmeteo_requests.Client:
    """Return an openmeteo_requests.Client using a cached & retrying session."""
    session = requests_cache.CachedSession(cache_name, expire_after=expire_after)
    session = retry(session, retries=retries, backoff_factor=backoff)
    client = openmeteo_requests.Client(session=session)
    return client


def fetch_openmeteo_archive(client: openmeteo_requests.Client,
                            latitude: float,
                            longitude: float,
                            start_date: str,
                            end_date: str,
                            which: str,   # "hourly" or "daily"
                            var: str      # variable name like "temperature_2m"
                            ) -> Tuple[Dict, pd.DataFrame]:
    """
    Fetch archive data and return (metadata_dict, dataframe).
    start_date/end_date must be ISO strings 'YYYY-MM-DD'.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        which: var,
    }
    responses = client.weather_api(url, params=params)
    response = responses[0]

    meta = {
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
        "elevation_m": response.Elevation(),
        "utc_offset_s": response.UtcOffsetSeconds(),
    }

    if which == "hourly":
        hourly = response.Hourly()
        vals = hourly.Variables(0).ValuesAsNumpy()
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        df = pd.DataFrame({"date_utc": times})
        df[var] = vals
    else:
        daily = response.Daily()
        vals = daily.Variables(0).ValuesAsNumpy()
        times = pd.to_datetime(daily.Time(), unit="s", utc=True)
        df = pd.DataFrame({"date_utc": times})
        df[var] = vals

    return meta, df
