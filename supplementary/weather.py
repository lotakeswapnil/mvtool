def fetch_openmeteo_archive(
        client: openmeteo_requests.Client,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        which: str,        # "hourly" or "daily"
        var: str,          # e.g. "temperature_2m"
        temp_unit: str = "celsius"  # NEW: "celsius" or "fahrenheit"
    ) -> Tuple[Dict, pd.DataFrame]:
    """
    Fetch archive data and return (metadata_dict, dataframe).
    start_date/end_date must be ISO strings 'YYYY-MM-DD'.
    temp_unit: "celsius" (default) or "fahrenheit".
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
        which: var,
    }

    responses = client.weather_api(url, params=params)
    response = responses[0]

    meta = {
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
        "elevation_m": response.Elevation(),
        "utc_offset_s": response.UtcOffsetSeconds(),
        "timezone": timezone,
        "temp_unit": temp_unit
    }

    # --- Helper: convert C → F ---
    def convert_temp(values):
        if temp_unit.lower() == "fahrenheit":
            return values * 9/5 + 32
        return values  # keep Celsius

    # --- Hourly data ---
    if which == "hourly":
        hourly = response.Hourly()
        vals = hourly.Variables(0).ValuesAsNumpy()

        # Convert temperatures only if var looks like a temperature field
        if "temp" in var or "temperature" in var:
            vals = convert_temp(vals)

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

    # --- Daily data ---
    else:
        daily = response.Daily()
        vals = daily.Variables(0).ValuesAsNumpy()

        if "temp" in var or "temperature" in var:
            vals = convert_temp(vals)

        times = pd.to_datetime(daily.Time(), unit="s", utc=True)
        df = pd.DataFrame({"date_utc": times})
        df[var] = vals

    return meta, df
