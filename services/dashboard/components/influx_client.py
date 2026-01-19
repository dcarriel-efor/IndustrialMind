"""InfluxDB client helper for querying sensor data."""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import streamlit as st
from influxdb_client import InfluxDBClient
import structlog

logger = structlog.get_logger("influx_client")


@st.cache_resource
def get_influx_client():
    """Get cached InfluxDB client connection.

    Returns:
        InfluxDBClient: Configured InfluxDB client
    """
    url = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
    token = os.getenv("INFLUXDB_TOKEN", "industrialmind-token-123456")
    org = os.getenv("INFLUXDB_ORG", "industrialmind")

    logger.info("connecting_to_influxdb", url=url, org=org)

    try:
        client = InfluxDBClient(url=url, token=token, org=org)
        logger.info("influxdb_client_created")
        return client
    except Exception as e:
        logger.error("failed_to_create_influx_client", error=str(e), error_type=type(e).__name__)
        raise


def get_available_machines() -> List[str]:
    """Get list of all machines with data in InfluxDB.

    Returns:
        List of machine IDs
    """
    logger.info("querying_available_machines")
    client = get_influx_client()
    org = os.getenv("INFLUXDB_ORG", "industrialmind")

    query = '''
    from(bucket: "sensors")
        |> range(start: -24h)
        |> filter(fn: (r) => r._measurement == "sensor_readings")
        |> keep(columns: ["machine_id"])
        |> distinct(column: "machine_id")
    '''

    try:
        logger.debug("executing_machines_query", query_preview=query[:100])
        query_api = client.query_api()
        result = query_api.query(query, org=org)

        machines = set()
        for table in result:
            for record in table.records:
                machines.add(record.values.get("machine_id"))

        logger.info("machines_queried", count=len(machines), machines=sorted(list(machines)))
        return sorted(list(machines))
    except Exception as e:
        logger.error("machines_query_failed", error=str(e), error_type=type(e).__name__, query_preview=query[:100])
        st.error(f"Error fetching machines: {str(e)}")
        return []


@st.cache_data(ttl=2)
def get_latest_readings(machine_id: str) -> Dict[str, float]:
    """Get the most recent sensor readings for a machine.

    Args:
        machine_id: Machine identifier

    Returns:
        Dictionary with latest sensor values
    """
    logger.info("fetching_latest_readings", machine_id=machine_id)
    client = get_influx_client()
    org = os.getenv("INFLUXDB_ORG", "industrialmind")

    query = f'''
    from(bucket: "sensors")
        |> range(start: -5m)
        |> filter(fn: (r) => r._measurement == "sensor_readings")
        |> filter(fn: (r) => r.machine_id == "{machine_id}")
        |> last()
    '''

    try:
        logger.debug("executing_latest_readings_query", machine_id=machine_id, query_preview=query[:100])
        query_api = client.query_api()
        result = query_api.query(query, org=org)

        readings = {}
        for table in result:
            for record in table.records:
                field = record.get_field()
                value = record.get_value()
                readings[field] = value

        logger.info("fetched_readings", machine_id=machine_id, field_count=len(readings), fields=list(readings.keys()))
        return readings
    except Exception as e:
        logger.error("latest_readings_query_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    machine_id=machine_id,
                    query_preview=query[:100])
        st.error(f"Error fetching latest readings: {str(e)}")
        return {}


@st.cache_data(ttl=5)
def get_time_series_data(
    machine_id: str,
    start_time: datetime,
    end_time: datetime,
    aggregation_window: str = "10s"
) -> pd.DataFrame:
    """Get time series data for a machine within a time range.

    Args:
        machine_id: Machine identifier
        start_time: Start of time range
        end_time: End of time range
        aggregation_window: Window for aggregation (e.g., "10s", "1m")

    Returns:
        DataFrame with timestamp and sensor values
    """
    logger.info("fetching_time_series",
               machine_id=machine_id,
               start=start_time.isoformat(),
               end=end_time.isoformat(),
               window=aggregation_window)
    client = get_influx_client()
    org = os.getenv("INFLUXDB_ORG", "industrialmind")

    # Format timestamps for InfluxDB query
    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    query = f'''
    from(bucket: "sensors")
        |> range(start: {start_str}, stop: {end_str})
        |> filter(fn: (r) => r._measurement == "sensor_readings")
        |> filter(fn: (r) => r.machine_id == "{machine_id}")
        |> aggregateWindow(every: {aggregation_window}, fn: mean, createEmpty: false)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''

    try:
        logger.debug("executing_time_series_query", machine_id=machine_id, query_preview=query[:150])
        query_api = client.query_api()
        result = query_api.query(query, org=org)

        # Convert to pandas DataFrame
        data = []
        for table in result:
            for record in table.records:
                row = {
                    "timestamp": record.get_time(),
                    "machine_id": record.values.get("machine_id"),
                    "sensor_id": record.values.get("sensor_id"),
                    "temperature": record.values.get("temperature"),
                    "vibration": record.values.get("vibration"),
                    "pressure": record.values.get("pressure"),
                    "power": record.values.get("power"),
                }
                data.append(row)

        if data:
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")
            logger.info("time_series_fetched", machine_id=machine_id, rows=len(df))
            return df
        else:
            logger.warning("no_time_series_data", machine_id=machine_id, start=start_str, end=end_str)
            return pd.DataFrame()

    except Exception as e:
        logger.error("time_series_query_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    machine_id=machine_id,
                    query_preview=query[:150])
        st.error(f"Error fetching time series data: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_machine_statistics(machine_id: str, hours: int = 24) -> Dict[str, Dict[str, float]]:
    """Get statistical summary for a machine over a time period.

    Args:
        machine_id: Machine identifier
        hours: Number of hours to look back

    Returns:
        Dictionary with stats (mean, min, max, stddev) for each sensor
    """
    client = get_influx_client()
    org = os.getenv("INFLUXDB_ORG", "industrialmind")

    query = f'''
    from(bucket: "sensors")
        |> range(start: -{hours}h)
        |> filter(fn: (r) => r._measurement == "sensor_readings")
        |> filter(fn: (r) => r.machine_id == "{machine_id}")
        |> group(columns: ["_field"])
    '''

    stats = {}

    for stat_name, stat_fn in [("mean", "mean"), ("min", "min"), ("max", "max"), ("stddev", "stddev")]:
        stat_query = query + f' |> {stat_fn}()'

        try:
            query_api = client.query_api()
            result = query_api.query(stat_query, org=org)

            for table in result:
                for record in table.records:
                    field = record.get_field()
                    value = record.get_value()

                    if field not in stats:
                        stats[field] = {}
                    stats[field][stat_name] = value

        except Exception as e:
            st.error(f"Error calculating {stat_name}: {str(e)}")

    return stats


def get_all_machines_latest() -> pd.DataFrame:
    """Get latest readings for all machines.

    Returns:
        DataFrame with latest readings for each machine
    """
    machines = get_available_machines()

    if not machines:
        return pd.DataFrame()

    data = []
    for machine_id in machines:
        readings = get_latest_readings(machine_id)
        if readings:
            readings["machine_id"] = machine_id
            data.append(readings)

    if data:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()
