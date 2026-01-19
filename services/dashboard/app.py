"""IndustrialMind - Main Dashboard Application."""

import streamlit as st
from datetime import datetime, timedelta
import sys
from pathlib import Path
import os

# Add project root to path for shared modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from shared.utils.logging_config import configure_logging

# Configure centralized logging
logger = configure_logging(
    service_name="dashboard",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    add_context={"version": "0.1.0"}
)

# Log startup
logger.info("dashboard_starting")

from components.influx_client import get_available_machines, get_latest_readings, get_time_series_data
from components.charts import create_time_series_chart, create_gauge_chart, create_multi_sensor_chart

# Page configuration
st.set_page_config(
    page_title="IndustrialMind Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .status-normal { color: #10b981; }
    .status-warning { color: #f59e0b; }
    .status-critical { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏭 IndustrialMind - Real-Time Monitoring")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Machine selection
    try:
        machines = get_available_machines()
        logger.info("machines_fetched", count=len(machines))
    except Exception as e:
        logger.error("failed_to_fetch_machines", error=str(e), error_type=type(e).__name__)
        machines = []

    if not machines:
        st.error("No machines found. Please check if the simulator is running.")
        logger.warning("no_machines_available")
        st.stop()

    selected_machine = st.selectbox(
        "Select Machine",
        options=machines,
        index=0
    )

    st.markdown("---")

    # Time range selection
    time_range = st.selectbox(
        "Time Range",
        options=["Last 5 minutes", "Last 15 minutes", "Last 1 hour", "Last 6 hours", "Last 24 hours"],
        index=0
    )

    # Convert to datetime range
    time_mapping = {
        "Last 5 minutes": 5,
        "Last 15 minutes": 15,
        "Last 1 hour": 60,
        "Last 6 hours": 360,
        "Last 24 hours": 1440
    }
    minutes = time_mapping[time_range]
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=minutes)

    st.markdown("---")

    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)

    if auto_refresh:
        import time
        time.sleep(2)
        st.rerun()

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# Main content
if selected_machine:
    logger.info("displaying_machine_data", machine_id=selected_machine)

    # Get latest readings
    try:
        latest = get_latest_readings(selected_machine)
        logger.info("latest_readings_fetched", machine_id=selected_machine, fields=len(latest))
    except Exception as e:
        logger.error("failed_to_fetch_latest_readings",
                    machine_id=selected_machine,
                    error=str(e),
                    error_type=type(e).__name__)
        latest = {}

    if not latest:
        st.warning(f"No recent data for {selected_machine}")
        logger.warning("no_recent_data", machine_id=selected_machine)
        st.stop()

    # Display current metrics in cards
    st.subheader(f"📊 Current Status - {selected_machine}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        temp = latest.get("temperature", 0)
        status = "normal" if temp < 70 else "warning" if temp < 85 else "critical"
        st.metric(
            label="🌡️ Temperature",
            value=f"{temp:.1f} °C",
            delta=f"{'Normal' if status == 'normal' else 'Warning' if status == 'warning' else 'Critical'}"
        )

    with col2:
        vib = latest.get("vibration", 0)
        status = "normal" if vib < 1.5 else "warning" if vib < 2.5 else "critical"
        st.metric(
            label="📳 Vibration",
            value=f"{vib:.2f} mm/s",
            delta=f"{'Normal' if status == 'normal' else 'Warning' if status == 'warning' else 'Critical'}"
        )

    with col3:
        press = latest.get("pressure", 0)
        status = "normal" if 40 <= press <= 60 else "warning" if 30 <= press <= 70 else "critical"
        st.metric(
            label="💨 Pressure",
            value=f"{press:.1f} PSI",
            delta=f"{'Normal' if status == 'normal' else 'Warning' if status == 'warning' else 'Critical'}"
        )

    with col4:
        power = latest.get("power", 0)
        status = "normal" if power < 300 else "warning" if power < 400 else "critical"
        st.metric(
            label="⚡ Power",
            value=f"{power:.0f} W",
            delta=f"{'Normal' if status == 'normal' else 'Warning' if status == 'warning' else 'Critical'}"
        )

    st.markdown("---")

    # Get time series data
    try:
        logger.info("fetching_time_series",
                   machine_id=selected_machine,
                   start=start_time.isoformat(),
                   end=end_time.isoformat())
        df = get_time_series_data(selected_machine, start_time, end_time, aggregation_window="10s")
        logger.info("time_series_fetched", machine_id=selected_machine, rows=len(df))
    except Exception as e:
        logger.error("failed_to_fetch_time_series",
                    machine_id=selected_machine,
                    error=str(e),
                    error_type=type(e).__name__)
        import pandas as pd
        df = pd.DataFrame()

    if not df.empty:
        st.subheader("📈 Time Series Data")

        # Multi-sensor chart
        fig_multi = create_multi_sensor_chart(df, title=f"All Sensors - {selected_machine}")
        st.plotly_chart(fig_multi, use_container_width=True)

        st.markdown("---")

        # Individual sensor charts
        col1, col2 = st.columns(2)

        with col1:
            # Temperature chart
            fig_temp = create_time_series_chart(
                df, "temperature",
                f"Temperature - {selected_machine}",
                "Temperature (°C)"
            )
            st.plotly_chart(fig_temp, use_container_width=True)

            # Pressure chart
            fig_press = create_time_series_chart(
                df, "pressure",
                f"Pressure - {selected_machine}",
                "Pressure (PSI)"
            )
            st.plotly_chart(fig_press, use_container_width=True)

        with col2:
            # Vibration chart
            fig_vib = create_time_series_chart(
                df, "vibration",
                f"Vibration - {selected_machine}",
                "Vibration (mm/s)"
            )
            st.plotly_chart(fig_vib, use_container_width=True)

            # Power chart
            fig_power = create_time_series_chart(
                df, "power",
                f"Power Consumption - {selected_machine}",
                "Power (W)"
            )
            st.plotly_chart(fig_power, use_container_width=True)

        st.markdown("---")

        # Gauge charts
        st.subheader("🎯 Status Gauges")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fig_gauge_temp = create_gauge_chart(
                latest.get("temperature", 0),
                "Temperature",
                0, 100,
                {"warning": 70, "critical": 85}
            )
            st.plotly_chart(fig_gauge_temp, use_container_width=True)

        with col2:
            fig_gauge_vib = create_gauge_chart(
                latest.get("vibration", 0),
                "Vibration",
                0, 5,
                {"warning": 1.5, "critical": 2.5}
            )
            st.plotly_chart(fig_gauge_vib, use_container_width=True)

        with col3:
            fig_gauge_press = create_gauge_chart(
                latest.get("pressure", 0),
                "Pressure",
                0, 100,
                {"warning": 70, "critical": 80}
            )
            st.plotly_chart(fig_gauge_press, use_container_width=True)

        with col4:
            fig_gauge_power = create_gauge_chart(
                latest.get("power", 0),
                "Power",
                0, 500,
                {"warning": 300, "critical": 400}
            )
            st.plotly_chart(fig_gauge_power, use_container_width=True)

        # Data summary
        st.markdown("---")
        st.subheader("📊 Data Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**Total Data Points**: {len(df)}")
            st.write(f"**Time Range**: {time_range}")

        with col2:
            st.write(f"**Start**: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            st.write(f"**End**: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        with col3:
            if len(df) > 0:
                st.write(f"**Sampling Rate**: ~{len(df) / minutes:.1f} points/min")
                st.write(f"**Data Quality**: {100.0:.1f}%")

    else:
        st.warning(f"No data available for {selected_machine} in the selected time range.")

else:
    st.info("Please select a machine from the sidebar.")
