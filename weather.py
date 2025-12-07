# weather_streamlit.py
import streamlit as st
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import date, timedelta
import io

st.set_page_config(page_title="Open-Meteo Archive Fetcher", layout="wide")
st.title("Open-Meteo Archive Fetcher")

# Sidebar inputs
st.sidebar.header("Query parameters")
lat = st.sidebar.number_input("Latitude", value=18.5196, format="%.6f")
lon = st.sidebar.number_input("Longitude", value=73.8554, format="%.6f")
today = date.today()
default_start = today - timedelta(days=365)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=today)
# ensure start <= end
if start_date > end_date:
    st.sidebar.error("Start date must be before or equal to end date")

# Hourly / daily variable choices (only temperature_2m shown as default; you can expand)
variable_options = {
    "temperature_2m (hourly)": ("hourly", "temperature_2m"),
    # add more if you need: "relativehumidity_2m (hourly)": ("hourly", "relativehumidity_2m"),
    # "shortwave_radiation_sum (daily)": ("daily", "shortwave_radiation_sum"),
}
var_label = st.sidebar.selectbox("Variable to request", list(variable_options.keys()))
which_data, var_name = variable_options[var_label]

# Caching + retry setup note
st.sidebar.markdown(
    """
**Network behavior**
- Uses `requests_cache` to cache responses locally in `.cache`.
- Retries up to 5 times on transient errors.
"""
)

# Prepare client (use cached session + retry wrapper)
@st.cache_data(show_spinner=False, persist="filesystem")
def make_openmeteo_client(cache_name=".cache", expire_after=-1, retries=5, backoff=0.2):
    session = requests_cache.CachedSession(cache_name, expire_after=expire_after)
    session = retry(session, retries=retries, backoff_factor=backoff)
    client = openmeteo_requests.Client(session=session)
    return client

client = make_openmeteo_client()

# Helper to fetch and convert response -> pandas.DataFrame
@st.cache_data(show_spinner=False)
def fetch_openmeteo_archive(client, latitude, longitude, start_date_str, end_date_str, which, var):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date_str,
        "end_date": end_date_str,
        which: var,
    }
    responses = client.weather_api(url, params=params)
    # We assume at least one response; process the first one
    response = responses[0]

    # metadata
    metadata = {
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
        "elevation_m": response.Elevation(),
        "utc_offset_s": response.UtcOffsetSeconds(),
    }

    # hourly/daily structure handling (mirrors original code)
    if which == "hourly":
        hourly = response.Hourly()
        # The order of Variables() corresponds to the requested variables; we requested a single one
        values = hourly.Variables(0).ValuesAsNumpy()
        # Build time index: start -> end exclusive using interval seconds
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        df = pd.DataFrame({"date_utc": times})
        df[var] = values
    else:
        # daily example (if you request daily variables)
        daily = response.Daily()
        values = daily.Variables(0).ValuesAsNumpy()
        times = pd.to_datetime(daily.Time(), unit="s", utc=True)
        df = pd.DataFrame({"date_utc": times})
        df[var] = values

    return metadata, df

# Main action
if st.button("Fetch Weather Data"):
    # validation
    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
    else:
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        with st.spinner("Fetching archived weather data... (this may take a moment)"):
            try:
                meta, df = fetch_openmeteo_archive(client, lat, lon, start_str, end_str, which_data, var_name)
            except Exception as e:
                st.error(f"Error fetching data: {e}")
            else:
                st.success("Data fetched successfully.")
                # show metadata
                st.write("**Location metadata:**")
                st.json(meta)
                # show dataframe
                st.subheader("Data preview")
                st.dataframe(df.head(200))

                # show simple time-series chart
                st.line_chart(df.set_index("date_utc")[var_name])

                # Provide CSV download
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                csv_bytes = csv_buf.getvalue().encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name=f"openmeteo_{lat}_{lon}_{start_str}_to_{end_str}.csv",
                    mime="text/csv",
                )

# Optionally show schema / example
st.markdown("---")
st.info("Tip: The `openmeteo-requests` client returns rich response objects. This app processes the first response and converts the requested variable to a pandas DataFrame. Modify the code to request multiple variables or multiple locations.")