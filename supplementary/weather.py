# weather_api.py
from io import StringIO

import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from typing import Tuple, Dict
from timezonefinder import TimezoneFinder
import requests


def make_openmeteo_client(cache_name: str = ".cache", expire_after: int = -1,
                          retries: int = 5, backoff: float = 0.2) -> openmeteo_requests.Client:
    """Return an openmeteo_requests.Client using a cached & retrying session."""
    session = requests_cache.CachedSession(cache_name, expire_after=expire_after)
    session = retry(session, retries=retries, backoff_factor=backoff)
    client = openmeteo_requests.Client(session=session)
    return client

# ---------------------------------------------
# TimeZone Function for Converting UTC to Local
# ---------------------------------------------

def get_timezone_from_coords(latitude: float, longitude: float) -> str:
    """Return IANA timezone name (e.g., 'Europe/Berlin') from coordinates."""
    tf = TimezoneFinder()
    tz = tf.timezone_at(lat=latitude, lng=longitude)
    if tz is None:
        raise ValueError(f"Could not determine timezone for coords {latitude}, {longitude}")
    return tz


# --------------------------
# OpenMeteo Weather Function
# --------------------------

def fetch_openmeteo_archive(client: openmeteo_requests.Client,
                            latitude: float,
                            longitude: float,
                            start_date: str,
                            end_date: str,
                            temperature_unit: str,
                            which: str,   # "hourly" or "daily"
                            var: str      # variable name like "temperature_2m"
                            ) -> Tuple[Dict, pd.DataFrame]:
    """
    Fetch archive data and return (metadata_dict, dataframe).
    start_date/end_date must be ISO strings 'YYYY-MM-DD'.
    """

    # --- Detect timezone ---
    timezone = get_timezone_from_coords(latitude, longitude)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "temperature_unit": temperature_unit,
        which: var,
    }
    responses = client.weather_api(url, params=params)
    response = responses[0]

    meta = {
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
        "elevation_m": response.Elevation(),
        "utc_offset_s": response.UtcOffsetSeconds(),
        "timezone": timezone
    }

    if which == "hourly":
        hourly = response.Hourly()
        vals = hourly.Variables(0).ValuesAsNumpy()
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s").tz_localize("UTC").tz_convert(timezone),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s").tz_localize("UTC").tz_convert(timezone),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        df = pd.DataFrame({
            "date_local": times,
            var: vals,
        })
    else:
        daily = response.Daily()
        vals = daily.Variables(0).ValuesAsNumpy()
        times = pd.to_datetime(daily.Time(), unit="s", utc=True)
        df = pd.DataFrame({"date_utc": times})
        df[var] = vals

    return meta, df


def pvgis_tmy(latitude, longitude):

    url = "https://re.jrc.ec.europa.eu/api/tmy"

    params = {
        "lat": latitude,
        "lon": longitude,
        "outputformat": "csv",
        "usehorizon": 1,
        "localtime": 1
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    lines = response.text.splitlines()

    # Find start of hourly data
    start_row = None
    for i, line in enumerate(lines):
        if line.startswith("time(UTC)"):
            start_row = i
            break

    if start_row is None:
        raise ValueError("Hourly TMY header not found")

    df = pd.read_csv(StringIO("\n".join(lines[start_row:start_row+8761])), sep=",", usecols=["time(UTC)", "T2m"])

    df['time(UTC)'] = pd.to_datetime(df['time(UTC)'], format='%Y%m%d:%H%M', utc=False)
    df.rename(columns={'time(UTC)':'Time','T2m': 'Temperature'}, inplace=True)

    return df