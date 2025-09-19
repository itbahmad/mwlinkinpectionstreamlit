import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import google.generativeai as genai
from sqlalchemy import create_engine
import io
import re
import folium
from folium.plugins import MarkerCluster
import leafmap.foliumap as leafmap
import math
import os

# Set page config
st.set_page_config(page_title="Data Assistant with Maps", layout="wide")

# ---------- HELPER FUNCTIONS FROM MICROWAVE INSPECTION ----------

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points (meters)"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def bearing(lat1, lon1, lat2, lon2):
    """Returns bearing in degrees from (lat1, lon1) to (lat2, lon2)"""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    diff_lon = np.radians(lon2 - lon1)
    x = np.sin(diff_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(diff_lon)
    initial_bearing = np.degrees(np.arctan2(x, y))
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def angle_diff(a1, a2):
    """Returns smallest difference between two angles (degrees)"""
    d = abs(a1 - a2) % 360
    return min(d, 360-d)

def estimate_beamwidth(freq):
    if 5925 <= freq <= 6425:   # 6 GHz
        return 4
    elif 7125 <= freq <= 8500: # 7/8 GHz
        return 3.5
    elif 10700 <= freq <= 11700: # 11 GHz
        return 2.5
    elif 12700 <= freq <= 13250: # 13 GHz
        return 2.5
    elif 14500 <= freq <= 15350: # 15 GHz
        return 2.5
    elif 17700 <= freq <= 19700: # 18 GHz
        return 1.5
    elif 21200 <= freq <= 23600: # 23 GHz
        return 1.5
    else:
        return 5  # Default wider beamwidth if unrecognized

def polarization_discrimination(plzn1, plzn2):
    """
    Returns the cross-polar discrimination factor (XPD in dB) and a boolean
    indicating whether polarization reduces interference.
    """
    if pd.isna(plzn1) or pd.isna(plzn2):
        return 0, False
    plzn1 = str(plzn1).upper()
    plzn2 = str(plzn2).upper()
    if plzn1 != plzn2 and plzn1 in ['H', 'V'] and plzn2 in ['H', 'V']:
        return 30, True  # typical XPD value (can adjust as needed)
    return 0, False

def analyze_conflicts(df, adjacent_thresh=14, spatial_thresh=5000, beam_overlap_margin=0.5):
    """
    Analyze potential frequency conflicts between stations.
    """
    results = []
    n = len(df)
    for i in range(n):
        for j in range(i+1, n):
            s1 = df.iloc[i]
            s2 = df.iloc[j]
            
            # Handle case-insensitive column names
            column_mapping = {col.lower(): col for col in df.columns}
            
            # Get frequency and bandwidth columns
            freq_col = next((column_mapping[name] for name in ['freq'] if name in column_mapping), None)
            bw_col = next((column_mapping[name] for name in ['bwidth'] if name in column_mapping), None)
            
            if not freq_col or not bw_col:
                continue
                
            freq1 = s1[freq_col]
            freq2 = s2[freq_col]
            bw1 = s1[bw_col]
            bw2 = s2[bw_col]
            
            kind = "Co-channel" if abs(freq1 - freq2) < 0.01 else "Adjacent channel" if abs(freq1 - freq2) < adjacent_thresh else None
            
            # Get lat/long columns using case-insensitive lookup
            lat_col = next((column_mapping[name] for name in ['sid_lat', 'latitude'] if name in column_mapping), None)
            lon_col = next((column_mapping[name] for name in ['sid_long', 'longitude'] if name in column_mapping), None)
            
            if not lat_col or not lon_col:
                continue
                
            dist = haversine(s1[lat_col], s1[lon_col], s2[lat_col], s2[lon_col])
            spatial_overlap = dist < spatial_thresh
            beam_overlap = False
            overlap_detail = ""
            polarization_effect = ""
            
            # Check for polarization column
            pol_col = next((column_mapping[name] for name in ['master_plzn_code', 'polarization'] if name in column_mapping), None)
            
            if pol_col:
                xpd_db, polz_reduced = polarization_discrimination(s1[pol_col], s2[pol_col])
            else:
                xpd_db, polz_reduced = 0, False
            
            if kind and spatial_overlap:
                # Check for azimuth column
                azim_col = next((column_mapping[name] for name in ['azimuth'] if name in column_mapping), None)
                beamwidth_col = next((column_mapping[name] for name in ['beamwidth'] if name in column_mapping), None)
                
                if azim_col:
                    try:
                        azim1 = float(s1[azim_col])
                        azim2 = float(s2[azim_col])
                        
                        if beamwidth_col and not pd.isna(s1.get(beamwidth_col, None)) and not pd.isna(s2.get(beamwidth_col, None)):
                            beamwidth1 = float(s1[beamwidth_col])
                            beamwidth2 = float(s2[beamwidth_col])
                        else:
                            beamwidth1 = estimate_beamwidth(freq1)
                            beamwidth2 = estimate_beamwidth(freq2)
                            
                        bearing12 = bearing(s1[lat_col], s1[lon_col], s2[lat_col], s2[lon_col])
                        bearing21 = bearing(s2[lat_col], s2[lon_col], s1[lat_col], s1[lon_col])
                        overlap1 = angle_diff(bearing12, azim1) <= beam_overlap_margin * beamwidth1
                        overlap2 = angle_diff(bearing21, azim2) <= beam_overlap_margin * beamwidth2
                        beam_overlap = overlap1 and overlap2
                        overlap_detail = (
                            f"Az1={azim1:.1f},Bear12={bearing12:.1f},Δ1={angle_diff(bearing12, azim1):.1f},BW1={beamwidth1:.1f}; "
                            f"Az2={azim2:.1f},Bear21={bearing21:.1f},Δ2={angle_diff(bearing21, azim2):.1f},BW2={beamwidth2:.1f}"
                        )
                    except Exception as e:
                        overlap_detail = f"Error: {e}"
                else:
                    overlap_detail = "No AZIMUTH info"
                
                if polz_reduced:
                    polarization_effect = f"Interference reduced by polarization (XPD: {xpd_db} dB)"
                else:
                    polarization_effect = "No polarization discrimination"
                
                # Get station names
                stn_name_col = next((column_mapping[name] for name in ['stn_name', 'station_name', 'name'] if name in column_mapping), None)
                link_id_col = next((column_mapping[name] for name in ['link_id'] if name in column_mapping), None)
                
                name1 = s1.get(stn_name_col, f"Station {i}") if stn_name_col else f"Station {i}"
                name2 = s2.get(stn_name_col, f"Station {j}") if stn_name_col else f"Station {j}"
                
                link_id1 = s1.get(link_id_col, '') if link_id_col else ''
                link_id2 = s2.get(link_id_col, '') if link_id_col else ''
                
                results.append({
                    'Station 1': name1,
                    'Station 2': name2,
                    'Freq 1 (MHz)': freq1,
                    'Freq 2 (MHz)': freq2,
                    'Bandwidth 1 (MHz)': bw1,
                    'Bandwidth 2 (MHz)': bw2,
                    'Type': kind,
                    'Distance (m)': round(dist,1),
                    'Beam Overlap': "Yes" if beam_overlap else "No",
                    'Overlap Detail': overlap_detail,
                    'Polarization Effect': polarization_effect,
                    'Link ID 1': link_id1,
                    'Link ID 2': link_id2,
                    'SID_LAT_1': s1[lat_col],
                    'SID_LONG_1': s1[lon_col],
                    'SID_LAT_2': s2[lat_col],
                    'SID_LONG_2': s2[lon_col]
                })
    return pd.DataFrame(results)

def create_map_for_conflicts_table(df, return_map=False):
    """Create a map specifically for a conflicts table that contains pairs of coordinates"""
    try:
        # Check if this is a conflicts table (has the specific columns)
        is_conflicts_table = all(col in df.columns for col in ['SID_LAT_1', 'SID_LONG_1', 'SID_LAT_2', 'SID_LONG_2'])
        
        if not is_conflicts_table:
            return "This doesn't appear to be a conflicts table with coordinate pairs"
        
        # Calculate map center from all coordinates
        lats = pd.concat([df['SID_LAT_1'], df['SID_LAT_2']]).dropna()
        longs = pd.concat([df['SID_LONG_1'], df['SID_LONG_2']]).dropna()
        
        if len(lats) == 0 or len(longs) == 0:
            return "No valid coordinates found in the conflicts table"
            
        center_lat = lats.astype(float).mean()
        center_long = longs.astype(float).mean()
        
        # Create map
        m = leafmap.Map(center=(center_lat, center_long), zoom=8)
        
        # Create a layer for stations
        stations_layer = folium.FeatureGroup(name="Stations")
        
        # Track unique stations to avoid duplicates
        unique_stations = {}
        
        # Create a layer for conflicts
        conflicts_layer = folium.FeatureGroup(name="Frequency Conflicts")
        
        # Add markers and lines for each conflict
        for idx, conflict in df.iterrows():
            # Get station coordinates
            lat1 = float(conflict['SID_LAT_1'])
            long1 = float(conflict['SID_LONG_1'])
            lat2 = float(conflict['SID_LAT_2'])
            long2 = float(conflict['SID_LONG_2'])
            
            # Skip if coordinates are not valid
            if np.isnan(lat1) or np.isnan(long1) or np.isnan(lat2) or np.isnan(long2):
                continue
                
            # Get station names
            station1 = conflict.get('Station 1', f"Station 1 - {idx}")
            station2 = conflict.get('Station 2', f"Station 2 - {idx}")
            
            # Add station 1 marker if not already added
            station_key1 = f"{lat1}_{long1}"
            if station_key1 not in unique_stations:
                unique_stations[station_key1] = True
                
                # Create a custom row for station 1 with relevant info
                station1_data = {
                    'SID_LAT': lat1,
                    'SID_LONG': long1,
                    'Freq': conflict.get('Freq 1 (MHz)', ''),
                    'Bandwidth': conflict.get('Bandwidth 1 (MHz)', ''),
                    'Link ID': conflict.get('Link ID 1', '')
                }
                
                # Create popup with scrollbar and Google Maps button
                popup_html1 = create_marker_popup(station1, pd.Series(station1_data))
                
                folium.Marker(
                    location=[lat1, long1],
                    popup=folium.Popup(popup_html1, max_width=300),
                    tooltip=station1,
                    icon=folium.Icon(color='blue', icon='antenna', prefix='fa')
                ).add_to(stations_layer)
            
            # Add station 2 marker if not already added
            station_key2 = f"{lat2}_{long2}"
            if station_key2 not in unique_stations:
                unique_stations[station_key2] = True
                
                # Create a custom row for station 2 with relevant info
                station2_data = {
                    'SID_LAT': lat2,
                    'SID_LONG': long2,
                    'Freq': conflict.get('Freq 2 (MHz)', ''),
                    'Bandwidth': conflict.get('Bandwidth 2 (MHz)', ''),
                    'Link ID': conflict.get('Link ID 2', '')
                }
                
                # Create popup with scrollbar and Google Maps button
                popup_html2 = create_marker_popup(station2, pd.Series(station2_data))
                
                folium.Marker(
                    location=[lat2, long2],
                    popup=folium.Popup(popup_html2, max_width=300),
                    tooltip=station2,
                    icon=folium.Icon(color='blue', icon='antenna', prefix='fa')
                ).add_to(stations_layer)
            
            # Create conflict details for popup with scrollbar
            conflict_popup = f"""
            <div style="max-height: 200px; overflow-y: auto; padding-right: 10px;">
                <h4>Potential Interference</h4>
                <b>Type:</b> {conflict.get('Type', 'Unknown')}<br>
                <b>Station 1:</b> {station1} ({conflict.get('Freq 1 (MHz)', '')} MHz)<br>
                <b>Station 2:</b> {station2} ({conflict.get('Freq 2 (MHz)', '')} MHz)<br>
                <b>Distance:</b> {conflict.get('Distance (m)', '')} meters<br>
                <b>Beam Overlap:</b> {conflict.get('Beam Overlap', '')}<br>
                <b>Polarization Effect:</b> {conflict.get('Polarization Effect', '')}<br>
                <b>Overlap Detail:</b> {conflict.get('Overlap Detail', '')}<br>
            </div>
            """
            
            # Draw dashed line between conflicting stations
            folium.PolyLine(
                locations=[[lat1, long1], [lat2, long2]],
                color='orange',  # Use a distinctive color for conflicts
                weight=3,
                opacity=0.8,
                dash_array='5, 10',  # Creates a dashed line
                popup=folium.Popup(conflict_popup, max_width=300),
                tooltip=f"Conflict: {station1} - {station2}"
            ).add_to(conflicts_layer)
        
        # Add layers to map
        stations_layer.add_to(m)
        conflicts_layer.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        if return_map:
            return m
        else:
            m.to_streamlit(height=600)
            return "Map displayed successfully"
    except Exception as e:
        return f"Error creating conflicts map: {e}"

def create_marker_popup(station_name, row, max_height=200):
    """Create a scrollable popup with a 'View in Google Maps' button"""
    
    # Start popup HTML with custom styling for scrollbar
    popup_html = f"""
    <div style="max-height: {max_height}px; overflow-y: auto; padding-right: 10px;">
        <h4>{station_name}</h4>
    """
    
    # Add other available information to popup (limited to most important columns)
    important_cols = ['CLNT_NAME', 'CURR_LIC_NUM', 'LINK_ID', 'FREQ', 'FREQ_PAIR', 
                      'BWIDTH', 'EQ_MDL', 'CITY', 'DISTRICT', 'AZIMUTH', 'POLARIZATION']
    
    # First add important columns
    for col in important_cols:
        if col in row and not pd.isna(row[col]):
            popup_html += f"<b>{col.replace('_', ' ')}:</b> {row[col]}<br>"
    
    # Then add other columns (excluding lat/long columns)
    for col in row.index:
        if (col not in important_cols and 
            col not in ['SID_LAT', 'SID_LONG', 'latitude', 'longitude'] and 
            not col.endswith('_lat') and not col.endswith('_long') and
            not pd.isna(row[col])):
            popup_html += f"<b>{col.replace('_', ' ')}:</b> {row[col]}<br>"
    
    # Close the scrollable div
    popup_html += "</div>"
    
    # Get coordinates for Google Maps link
    lat = None
    lon = None
    
    # Try to find coordinates from various possible column names
    for lat_col in ['SID_LAT', 'latitude', 'SID_LAT_1']:
        if lat_col in row and not pd.isna(row[lat_col]):
            lat = row[lat_col]
            break
    
    for lon_col in ['SID_LONG', 'longitude', 'SID_LONG_1']:
        if lon_col in row and not pd.isna(row[lon_col]):
            lon = row[lon_col]
            break
    
    # Add Google Maps button if coordinates are available
    if lat is not None and lon is not None:
        popup_html += f"""
        <div style="margin-top: 10px; text-align: center;">
            <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank" 
               style="text-decoration: none;">
                <button style="background-color: #4285F4; color: white; border: none; 
                        padding: 5px 10px; border-radius: 3px; cursor: pointer;">
                    Open in Google Maps
                </button>
            </a>
        </div>
        """
    
    return popup_html

def create_combined_map(dataframes, return_map=False):
    """
    Create a combined map with multiple data layers
    
    Parameters:
    -----------
    dataframes : dict
        Dictionary with table names as keys and dataframes as values
    
    return_map : bool
        Whether to return the map object or display it directly
    """
    try:
        # First, calculate map center from all dataframes
        all_lats = []
        all_longs = []
        
        # Track all valid coordinates
        for table_name, df in dataframes.items():
            # Check if this is a conflicts table
            is_conflicts_table = all(col in df.columns for col in 
                                    ['SID_LAT_1', 'SID_LONG_1', 'SID_LAT_2', 'SID_LONG_2'])
            
            if is_conflicts_table:
                lats = pd.concat([df['SID_LAT_1'], df['SID_LAT_2']]).dropna()
                longs = pd.concat([df['SID_LONG_1'], df['SID_LONG_2']]).dropna()
                all_lats.extend(lats.tolist())
                all_longs.extend(longs.tolist())
            else:
                # Handle case-insensitive column names
                column_mapping = {col.lower(): col for col in df.columns}
                
                # Get lat/long columns using case-insensitive lookup
                lat_col = next((column_mapping[name] for name in ['sid_lat', 'latitude'] 
                              if name in column_mapping), None)
                lon_col = next((column_mapping[name] for name in ['sid_long', 'longitude'] 
                              if name in column_mapping), None)
                
                if lat_col and lon_col:
                    lats = df[lat_col].dropna()
                    longs = df[lon_col].dropna()
                    all_lats.extend(lats.tolist())
                    all_longs.extend(longs.tolist())
        
        if not all_lats or not all_longs:
            return "No valid coordinates found in any of the tables"
            
        center_lat = sum(all_lats) / len(all_lats)
        center_long = sum(all_longs) / len(all_longs)
        
        # Create base map
        m = leafmap.Map(center=(center_lat, center_long), zoom=8)
        
        # Track unique stations to avoid duplicates
        unique_stations = {}
        
        # Process each dataframe
        for table_name, df in dataframes.items():
            # Check if this is a conflicts table
            is_conflicts_table = all(col in df.columns for col in 
                                    ['SID_LAT_1', 'SID_LONG_1', 'SID_LAT_2', 'SID_LONG_2'])
            
            if is_conflicts_table:
                # Create layers for conflicts table
                stations_layer = folium.FeatureGroup(name=f"Stations ({table_name})")
                conflicts_layer = folium.FeatureGroup(name=f"Conflicts ({table_name})")
                
                # Add markers and lines for each conflict
                for idx, conflict in df.iterrows():
                    # Get station coordinates
                    try:
                        lat1 = float(conflict['SID_LAT_1'])
                        long1 = float(conflict['SID_LONG_1'])
                        lat2 = float(conflict['SID_LAT_2'])
                        long2 = float(conflict['SID_LONG_2'])
                        
                        # Skip if coordinates are not valid
                        if np.isnan(lat1) or np.isnan(long1) or np.isnan(lat2) or np.isnan(long2):
                            continue
                            
                        # Get station names
                        station1 = conflict.get('Station 1', f"Station 1 - {idx}")
                        station2 = conflict.get('Station 2', f"Station 2 - {idx}")
                        
                        # Add station 1 marker if not already added
                        station_key1 = f"{lat1}_{long1}"
                        if station_key1 not in unique_stations:
                            unique_stations[station_key1] = True
                            popup_html1 = f"<h4>{station1}</h4>"
                            
                            # Add other available information to popup
                            for col in ['Freq 1 (MHz)', 'Bandwidth 1 (MHz)', 'Link ID 1']:
                                if col in conflict and not pd.isna(conflict[col]):
                                    popup_html1 += f"<b>{col}:</b> {conflict[col]}<br>"
                            
                            folium.Marker(
                                location=[lat1, long1],
                                popup=folium.Popup(popup_html1, max_width=300),
                                tooltip=station1,
                                icon=folium.Icon(color='blue', icon='antenna', prefix='fa')
                            ).add_to(stations_layer)
                        
                        # Add station 2 marker if not already added
                        station_key2 = f"{lat2}_{long2}"
                        if station_key2 not in unique_stations:
                            unique_stations[station_key2] = True
                            popup_html2 = f"<h4>{station2}</h4>"
                            
                            # Add other available information to popup
                            for col in ['Freq 2 (MHz)', 'Bandwidth 2 (MHz)', 'Link ID 2']:
                                if col in conflict and not pd.isna(conflict[col]):
                                    popup_html2 += f"<b>{col}:</b> {conflict[col]}<br>"
                            
                            folium.Marker(
                                location=[lat2, long2],
                                popup=folium.Popup(popup_html2, max_width=300),
                                tooltip=station2,
                                icon=folium.Icon(color='blue', icon='antenna', prefix='fa')
                            ).add_to(stations_layer)
                        
                        # Create conflict details for popup
                        conflict_popup = f"""
                        <h4>Potential Interference</h4>
                        <b>Type:</b> {conflict.get('Type', 'Unknown')}<br>
                        <b>Station 1:</b> {station1} ({conflict.get('Freq 1 (MHz)', '')} MHz)<br>
                        <b>Station 2:</b> {station2} ({conflict.get('Freq 2 (MHz)', '')} MHz)<br>
                        <b>Distance:</b> {conflict.get('Distance (m)', '')} meters<br>
                        <b>Beam Overlap:</b> {conflict.get('Beam Overlap', '')}<br>
                        <b>Polarization Effect:</b> {conflict.get('Polarization Effect', '')}<br>
                        """
                        
                        # Draw dashed line between conflicting stations
                        folium.PolyLine(
                            locations=[[lat1, long1], [lat2, long2]],
                            color='orange',  # Use a distinctive color for conflicts
                            weight=3,
                            opacity=0.8,
                            dash_array='5, 10',  # Creates a dashed line
                            popup=folium.Popup(conflict_popup, max_width=300),
                            tooltip=f"Conflict: {station1} - {station2}"
                        ).add_to(conflicts_layer)
                    except Exception as e:
                        st.warning(f"Error processing conflict {idx}: {e}")
                
                # Add layers to map
                stations_layer.add_to(m)
                conflicts_layer.add_to(m)
                
            else:
                # Regular table
                # Get the latitude and longitude columns
                column_mapping = {col.lower(): col for col in df.columns}
                
                lat_col = next((column_mapping[name] for name in ['sid_lat', 'latitude'] 
                              if name in column_mapping), None)
                lon_col = next((column_mapping[name] for name in ['sid_long', 'longitude'] 
                              if name in column_mapping), None)
                
                if not lat_col or not lon_col:
                    continue  # Skip this table if no lat/long columns
                
                # Create station marker layer
                station_layer = folium.FeatureGroup(name=f"Stations ({table_name})")
                
                # Create links layer
                links_layer = folium.FeatureGroup(name=f"Links ({table_name})")
                
                # Get station name and link ID columns
                stn_name_col = next((column_mapping[name] for name in ['stn_name', 'station_name', 'name'] 
                                  if name in column_mapping), None)
                link_id_col = next((column_mapping[name] for name in ['link_id'] 
                                 if name in column_mapping), None)
                
                # Track stations by link
                stations_by_link = {}
                
                # Add markers for each station
                for idx, row in df.iterrows():
                    try:
                        lat = float(row[lat_col])
                        long = float(row[lon_col])
                        
                        if np.isnan(lat) or np.isnan(long):
                            continue
                        
                        # Get station name
                        station_name = row.get(stn_name_col, f"Station {idx}") if stn_name_col else f"Station {idx}"
                        
                        # Create popup HTML based on available columns
                        popup_html = "<h4>{}</h4>".format(station_name)
                        
                        # Add other available information to popup
                        for col in df.columns:
                            if col not in [lat_col, lon_col, stn_name_col] and not pd.isna(row[col]):
                                popup_html += f"<b>{col.replace('_', ' ')}:</b> {row[col]}<br>"
                        
                        # Create a unique key for this station
                        station_key = f"{lat}_{long}"
                        
                        # Only add if we haven't already added this station
                        if station_key not in unique_stations:
                            unique_stations[station_key] = True
                            
                            folium.Marker(
                                location=[lat, long],
                                popup=folium.Popup(popup_html, max_width=300),
                                tooltip=station_name,
                                icon=folium.Icon(color='blue', icon='antenna', prefix='fa')
                            ).add_to(station_layer)
                        
                        # Store stations by link ID for drawing lines
                        if link_id_col:
                            link_id = row.get(link_id_col)
                            if link_id and link_id != '':
                                if link_id not in stations_by_link:
                                    stations_by_link[link_id] = []
                                stations_by_link[link_id].append({
                                    'lat': lat, 
                                    'long': long, 
                                    'name': station_name,
                                    'far_end': row.get('STASIUN_LAWAN', '') if 'STASIUN_LAWAN' in row else ''
                                })
                    except Exception as e:
                        st.warning(f"Error processing station {idx}: {e}")
                
                # Draw lines between connected stations
                for link_id, stations in stations_by_link.items():
                    if len(stations) == 2:
                        station1 = stations[0]
                        station2 = stations[1]
                        link_popup = f"""
                        <h4>Link ID: {link_id}</h4>
                        <b>Station 1:</b> {station1['name']}<br>
                        <b>Station 2:</b> {station2['name']}<br>
                        """
                        folium.PolyLine(
                            locations=[[station1['lat'], station1['long']], [station2['lat'], station2['long']]],
                            color='red', weight=2, opacity=0.7,
                            popup=folium.Popup(link_popup, max_width=300)
                        ).add_to(links_layer)
                
                # Add layers to map
                station_layer.add_to(m)
                links_layer.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        if return_map:
            return m
        else:
            m.to_streamlit(height=600)
            return "Map displayed successfully"
    except Exception as e:
        return f"Error creating combined map: {e}"

def create_map_with_conflicts(df, conflicts_df=None, return_map=False):
    """Create a folium map with markers for stations, lines for links, and highlight conflicts"""
    try:
        # Handle case-insensitive column names
        column_mapping = {col.lower(): col for col in df.columns}
        
        # Get lat/long columns using case-insensitive lookup
        lat_col = next((column_mapping[name] for name in ['sid_lat', 'latitude'] if name in column_mapping), None)
        lon_col = next((column_mapping[name] for name in ['sid_long', 'longitude'] if name in column_mapping), None)
        
        if not lat_col or not lon_col:
            return "Map could not be created: Missing latitude/longitude columns"
        
        # Calculate map center
        center_lat = df[lat_col].astype(float).mean()
        center_long = df[lon_col].astype(float).mean()
        
        # Create map
        m = leafmap.Map(center=(center_lat, center_long), zoom=8)
        marker_cluster = MarkerCluster().add_to(m)
        stations_by_link = {}
        
        # Station name to coordinates mapping (for conflict lines later)
        station_coords = {}
        
        # Get station name column
        stn_name_col = next((column_mapping[name] for name in ['stn_name', 'station_name', 'name'] if name in column_mapping), None)
        link_id_col = next((column_mapping[name] for name in ['link_id'] if name in column_mapping), None)
        
        # Add markers for each station
        for idx, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                long = float(row[lon_col])
                
                if np.isnan(lat) or np.isnan(long):
                    continue
                
                # Get station name
                station_name = row.get(stn_name_col, f"Station {idx}") if stn_name_col else f"Station {idx}"
                
                # Store station coordinates for conflict lines
                station_coords[station_name] = (lat, long)
                
                # Create popup HTML based on available columns
                popup_html = "<h4>{}</h4>".format(station_name)
                
                # Add other available information to popup
                for col in df.columns:
                    if col not in [lat_col, lon_col, stn_name_col] and not pd.isna(row[col]):
                        popup_html += f"<b>{col.replace('_', ' ')}:</b> {row[col]}<br>"
                
                folium.Marker(
                    location=[lat, long],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=station_name,
                    icon=folium.Icon(color='blue', icon='antenna', prefix='fa')
                ).add_to(marker_cluster)
                
                # Store stations by link ID for drawing lines
                if link_id_col:
                    link_id = row.get(link_id_col)
                    if link_id and link_id != '':
                        if link_id not in stations_by_link:
                            stations_by_link[link_id] = []
                        stations_by_link[link_id].append({
                            'lat': lat, 
                            'long': long, 
                            'name': station_name,
                            'far_end': row.get('STASIUN_LAWAN', '') if 'STASIUN_LAWAN' in row else ''
                        })
            except Exception as e:
                st.warning(f"Could not add marker for station {idx}: {e}")
        
        # Draw lines between connected stations
        for link_id, stations in stations_by_link.items():
            if len(stations) == 2:
                station1 = stations[0]
                station2 = stations[1]
                link_popup = f"""
                <h4>Link ID: {link_id}</h4>
                <b>Station 1:</b> {station1['name']}<br>
                <b>Station 2:</b> {station2['name']}<br>
                """
                folium.PolyLine(
                    locations=[[station1['lat'], station1['long']], [station2['lat'], station2['long']]],
                    color='red', weight=2, opacity=0.7,
                    popup=folium.Popup(link_popup, max_width=300)
                ).add_to(m)
        
        # Draw conflict lines if conflicts_df is provided
        if conflicts_df is not None and not conflicts_df.empty:
            # Create a conflict layer
            conflict_layer = folium.FeatureGroup(name="Frequency Conflicts")
            
            for idx, conflict in conflicts_df.iterrows():
                # Check if we have direct coordinates in the conflicts dataframe
                if all(col in conflict for col in ['SID_LAT_1', 'SID_LONG_1', 'SID_LAT_2', 'SID_LONG_2']):
                    # Use direct coordinates from conflicts dataframe
                    loc1 = (conflict['SID_LAT_1'], conflict['SID_LONG_1'])
                    loc2 = (conflict['SID_LAT_2'], conflict['SID_LONG_2'])
                    station1 = conflict.get('Station 1', f"Station 1 - {idx}")
                    station2 = conflict.get('Station 2', f"Station 2 - {idx}")
                elif 'Station 1' in conflict and 'Station 2' in conflict:
                    # Look up coordinates from station_coords dictionary
                    station1 = conflict['Station 1']
                    station2 = conflict['Station 2']
                    
                    if station1 not in station_coords or station2 not in station_coords:
                        continue
                        
                    loc1 = station_coords[station1]
                    loc2 = station_coords[station2]
                else:
                    continue
                
                # Create conflict details for popup
                conflict_popup = f"""
                <h4>Potential Interference</h4>
                <b>Type:</b> {conflict.get('Type', 'Unknown')}<br>
                <b>Station 1:</b> {station1} ({conflict.get('Freq 1 (MHz)', '')} MHz)<br>
                <b>Station 2:</b> {station2} ({conflict.get('Freq 2 (MHz)', '')} MHz)<br>
                <b>Distance:</b> {conflict.get('Distance (m)', '')} meters<br>
                <b>Beam Overlap:</b> {conflict.get('Beam Overlap', '')}<br>
                <b>Polarization Effect:</b> {conflict.get('Polarization Effect', '')}<br>
                """
                
                # Draw dashed line between conflicting stations
                folium.PolyLine(
                    locations=[loc1, loc2],
                    color='orange',  # Use a distinctive color for conflicts
                    weight=3,
                    opacity=0.8,
                    dash_array='5, 10',  # Creates a dashed line
                    popup=folium.Popup(conflict_popup, max_width=300),
                    tooltip=f"Conflict: {station1} - {station2}"
                ).add_to(conflict_layer)
            
            conflict_layer.add_to(m)
            
            # Add layer control to toggle conflicts visibility
            folium.LayerControl().add_to(m)
        
        if return_map:
            return m
        else:
            m.to_streamlit(height=600)
            return "Map displayed successfully"
    except Exception as e:
        return f"Error creating map: {e}"

def create_map(df, return_map=False):
    """Create a folium map with markers for stations and lines for links"""
    try:
        # Handle case-insensitive column names
        column_mapping = {col.lower(): col for col in df.columns}
        
        # Get lat/long columns using case-insensitive lookup
        lat_col = next((column_mapping[name] for name in ['sid_lat', 'latitude'] if name in column_mapping), None)
        lon_col = next((column_mapping[name] for name in ['sid_long', 'longitude'] if name in column_mapping), None)
        
        if not lat_col or not lon_col:
            return "Map could not be created: Missing latitude/longitude columns"
        
        # Calculate map center
        center_lat = df[lat_col].astype(float).mean()
        center_long = df[lon_col].astype(float).mean()
        
        # Create map
        m = leafmap.Map(center=(center_lat, center_long), zoom=8)
        marker_cluster = MarkerCluster().add_to(m)
        stations_by_link = {}
        
        # Get station name column
        stn_name_col = next((column_mapping[name] for name in ['stn_name', 'station_name', 'name'] 
                          if name in column_mapping), None)
        link_id_col = next((column_mapping[name] for name in ['link_id'] if name in column_mapping), None)
        
        # Add markers for each station
        for idx, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                long = float(row[lon_col])
                
                if np.isnan(lat) or np.isnan(long):
                    continue
                
                # Get station name
                station_name = row.get(stn_name_col, f"Station {idx}") if stn_name_col else f"Station {idx}"
                
                # Create popup with scrollbar and Google Maps button
                popup_html = create_marker_popup(station_name, row)
                
                folium.Marker(
                    location=[lat, long],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=station_name,
                    icon=folium.Icon(color='blue', icon='antenna', prefix='fa')
                ).add_to(marker_cluster)
                
                # Store stations by link ID for drawing lines
                if link_id_col:
                    link_id = row.get(link_id_col)
                    if link_id and link_id != '':
                        if link_id not in stations_by_link:
                            stations_by_link[link_id] = []
                        stations_by_link[link_id].append({
                            'lat': lat, 
                            'long': long, 
                            'name': station_name,
                            'far_end': row.get('STASIUN_LAWAN', '') if 'STASIUN_LAWAN' in row else ''
                        })
            except Exception as e:
                pass
        
        # Draw lines between connected stations
        for link_id, stations in stations_by_link.items():
            if len(stations) == 2:
                station1 = stations[0]
                station2 = stations[1]
                link_popup = f"""
                <h4>Link ID: {link_id}</h4>
                <b>Station 1:</b> {station1['name']}<br>
                <b>Station 2:</b> {station2['name']}<br>
                """
                folium.PolyLine(
                    locations=[[station1['lat'], station1['long']], [station2['lat'], station2['long']]],
                    color='red', weight=2, opacity=0.7,
                    popup=folium.Popup(link_popup, max_width=300)
                ).add_to(m)
        
        if return_map:
            return m
        else:
            m.to_streamlit(height=600)
            return "Map displayed successfully"
    except Exception as e:
        return f"Error creating map: {e}"
    
def save_conflicts_to_table(conflicts_df, table_name, conn, engine):
    """Create a new table for conflicts with geographic information"""
    try:
        # Save conflicts dataframe to database
        save_dataframe_to_postgres(conflicts_df, table_name, engine)
        
        return True, f"Saved {len(conflicts_df)} conflicts to table '{table_name}'"
    except Exception as e:
        return False, f"Error saving conflicts to table: {e}"

def create_relations(license_df, inspection_df):
    """Create relations between license and inspection data"""
    merged_df = pd.merge(
        license_df, 
        inspection_df, 
        on='LINK_ID', 
        how='left', 
        suffixes=('', '_inspection')
    )
    
    # Update frequency pairs for each link
    link_groups = license_df.groupby('LINK_ID')
    updated_df = license_df.copy()
    
    for link_id, group in link_groups:
        if len(group) == 2:
            stations = group.index.tolist()
            for i in range(2):
                j = 1 - i
                if pd.isna(updated_df.loc[stations[i], 'FREQ_PAIR']) or updated_df.loc[stations[i], 'FREQ_PAIR'] == '':
                    updated_df.loc[stations[i], 'FREQ_PAIR'] = updated_df.loc[stations[j], 'FREQ']
    
    return updated_df, merged_df

# ---------- DATABASE FUNCTIONS ----------

def connect_to_db():
    try:
        # Get database connection parameters from environment variables or Streamlit secrets
        db_host = st.secrets.get("DB_HOST", "localhost")
        db_name = st.secrets.get("DB_NAME", "postgres")
        db_user = st.secrets.get("DB_USER", "postgres")
        db_password = st.secrets.get("DB_PASSWORD", "postgres")
        db_port = st.secrets.get("DB_PORT", "5432")
        
        # Create SQLAlchemy engine for pandas to_sql operations
        engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
        
        # Create direct connection for SQL queries
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        
        return conn, engine
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None

def save_dataframe_to_postgres(df, table_name, engine):
    try:
        # Save dataframe to PostgreSQL without modifying column names
        # This preserves the case of column names
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data to PostgreSQL: {e}")
        return False

def get_tables(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [table[0] for table in cursor.fetchall()]
        cursor.close()
        return tables
    except Exception as e:
        st.error(f"Error fetching tables: {e}")
        return []

def get_table_schema(conn, table_name):
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """)
        schema = cursor.fetchall()
        cursor.close()
        return schema
    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return []

def execute_query(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        
        # If the query is a SELECT query, return the results
        if query.strip().lower().startswith('select'):
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            return pd.DataFrame(results, columns=columns)
        else:
            # For non-SELECT queries, commit the changes and return success message
            conn.commit()
            cursor.close()
            return "Query executed successfully"
    except Exception as e:
        st.error(f"Query execution error: {e}")
        return f"Error: {e}"

# ---------- GEMINI AI FUNCTIONS ----------

def initialize_gemini():
    try:
        # Get API key from environment variables or Streamlit secrets
        api_key = st.secrets.get("GEMINI_API_KEY", "your-api-key")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.error(f"Gemini API initialization error: {e}")
        return None

def calculate(text):
    # Detect calculations using regex
    calculation_pattern = r'\bcalculate\s+([0-9+\-*/.\(\)\s]+)'
    matches = re.findall(calculation_pattern, text, re.IGNORECASE)
    
    if matches:
        try:
            for expr in matches:
                result = eval(expr)
                text = text.replace(f"calculate {expr}", f"{result}")
            return text
        except Exception as e:
            return f"Error in calculation: {e}"
    return text

# ---------- MAIN APPLICATION ----------

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}

if 'conflicts_data' not in st.session_state:
    st.session_state.conflicts_data = {}

def main():
    st.title("Smart Microwave Link Data Analyzer")
    
    # Initialize database connection and Gemini model
    conn, engine = connect_to_db()
    model = initialize_gemini()
    
    # Sidebar for uploading files and database management
    with st.sidebar:
        st.header("Upload Data")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls", "csv"])
        
        if uploaded_file is not None:
            # Read file
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.write("File preview:")
                st.dataframe(df.head())
                
                # Ask for table name
                table_name = st.text_input("Enter table name for this data:")
                
                if table_name and st.button("Save to Database"):
                    if conn and engine:
                        if save_dataframe_to_postgres(df, table_name, engine):
                            st.success(f"Data saved to table '{table_name}' successfully!")
                            # Also store in session state
                            st.session_state.dataframes[table_name] = df
                        else:
                            st.error("Failed to save data to database.")
                    else:
                        st.warning("Database connection not available. Storing in session only.")
                        st.session_state.dataframes[table_name] = df
                        st.success(f"Data saved to session as '{table_name}'")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Display database tables
        st.header("Database Tables")
        
        # Display tables from database
        if conn:
            tables = get_tables(conn)
            if tables:
                st.write("Database Tables:")
                for table in tables:
                    st.write(f"- {table}")
            else:
                st.write("No tables found in the database.")
        
        # Display tables from session state
        if st.session_state.dataframes:
            st.write("Session Tables:")
            for table in st.session_state.dataframes.keys():
                st.write(f"- {table} (session)")
    
    # Main chat interface
    st.header("Chat with your Data")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # If there's a dataframe result to display
            if "dataframe" in message:
                st.dataframe(message["dataframe"])
            
            # If there's a map to display
            if "map" in message and message["map"]:
                if isinstance(message["map"], str):
                    st.info(message["map"])
                else:
                    try:
                        # Try to display the map in Streamlit
                        message["map"].to_streamlit(height=600)
                    except Exception as e:
                        st.error(f"Could not display map: {e}")
    
    # Chat input
    user_input = st.chat_input("Ask me about your data...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process the query
        tables_info = ""
        
        # Get tables from database
        db_tables = []
        if conn:
            db_tables = get_tables(conn)
            for table in db_tables:
                schema = get_table_schema(conn, table)
                columns = ", ".join([f"{col} ({dtype})" for col, dtype in schema])
                tables_info += f"Table: {table}, Columns: {columns}\n"
        
        # Get tables from session state
        for table_name, df in st.session_state.dataframes.items():
            columns = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
            tables_info += f"Table: {table_name} (session), Columns: {columns}\n"
        
        # Check if the query contains a calculation request
        processed_query = calculate(user_input)
        
        # Special command handling
        response_content = ""
        response_df = None
        response_map = None
        
        # Check for map creation request
        if re.search(r'\b(create|show|display|generate|buat|tampilkan|lihat)\s+(?:peta|map)\b', user_input.lower()):
            map_table_match = re.search(r'(?:peta|map)\s+(?:of|for|from|dari|untuk)\s+(\w+)', user_input.lower())
            table_to_map = None
            
            if map_table_match:
                table_to_map = map_table_match.group(1)
            elif len(st.session_state.dataframes) == 1:
                table_to_map = list(st.session_state.dataframes.keys())[0]
            
            if table_to_map:
                # Get the table data
                table_exists = False
                df_to_map = None
                
                if table_to_map in st.session_state.dataframes:
                    df_to_map = st.session_state.dataframes[table_to_map]
                    table_exists = True
                elif conn and table_to_map in db_tables:
                    # Fetch the table from database
                    df_to_map = execute_query(conn, f"SELECT * FROM {table_to_map}")
                    if isinstance(df_to_map, pd.DataFrame):
                        table_exists = True
                
                if table_exists and df_to_map is not None:
                    response_content = f"Creating map for table '{table_to_map}'...\n\n"
                    
                    # Check if this is a conflicts table
                    is_conflicts_table = all(col in df_to_map.columns for col in 
                                            ['SID_LAT_1', 'SID_LONG_1', 'SID_LAT_2', 'SID_LONG_2'])
                    
                    if is_conflicts_table:
                        # Use the specialized function for conflicts tables
                        response_map = create_map_for_conflicts_table(df_to_map, return_map=True)
                    else:
                        # Use the regular map creation function
                        response_map = create_map(df_to_map, return_map=True)
                else:
                    response_content = f"Table '{table_to_map}' not found."
            else:
                response_content = "Please specify which table to create a map for."

        # Check for combined map creation request
        elif re.search(r'\b(show|display|tampilkan|lihat)\s+(?:peta|map)\s+(?:for|dari|untuk)\s+(.*?)\s+(?:and|dan)\s+(.*?)\s+(?:table|tables|tabel)', user_input.lower()):
            # Extract table names
            map_tables_match = re.search(r'(?:peta|map)\s+(?:for|dari|untuk)\s+(.*?)\s+(?:and|dan)\s+(.*?)\s+(?:table|tables|tabel)', user_input.lower())
            
            if map_tables_match:
                table1 = map_tables_match.group(1).strip()
                table2 = map_tables_match.group(2).strip()
                
                # Get the dataframes
                dataframes_to_map = {}
                
                # Check for first table
                if table1 in st.session_state.dataframes:
                    dataframes_to_map[table1] = st.session_state.dataframes[table1]
                elif conn and table1 in db_tables:
                    df1 = execute_query(conn, f"SELECT * FROM {table1}")
                    if isinstance(df1, pd.DataFrame):
                        dataframes_to_map[table1] = df1
                
                # Check for second table
                if table2 in st.session_state.dataframes:
                    dataframes_to_map[table2] = st.session_state.dataframes[table2]
                elif conn and table2 in db_tables:
                    df2 = execute_query(conn, f"SELECT * FROM {table2}")
                    if isinstance(df2, pd.DataFrame):
                        dataframes_to_map[table2] = df2
                
                if dataframes_to_map:
                    response_content = f"Creating combined map for tables: {', '.join(dataframes_to_map.keys())}...\n\n"
                    response_map = create_combined_map(dataframes_to_map, return_map=True)
                else:
                    response_content = "Could not find the specified tables."
            else:
                response_content = "Please specify which tables to include in the map."
                

        # Check for conflict analysis request
        elif re.search(r'\b(analyze|check|find|analisa|cek|temukan|periksa)\s+(conflicts|interference|konflik|interferensi)\b', user_input.lower()):
            conflict_table_match = re.search(r'(conflicts|interference|konflik|interferensi)\s+(?:in|for|from|pada|di|dari|untuk)\s+(\w+)', user_input.lower())
            table_to_analyze = None
            
            if conflict_table_match:
                table_to_analyze = conflict_table_match.group(2)
            elif len(st.session_state.dataframes) == 1:
                table_to_analyze = list(st.session_state.dataframes.keys())[0]
            
            if table_to_analyze:
                # Check if the table exists
                table_exists = False
                df_to_analyze = None
                
                if table_to_analyze in st.session_state.dataframes:
                    df_to_analyze = st.session_state.dataframes[table_to_analyze]
                    table_exists = True
                elif conn and table_to_analyze in db_tables:
                    # Fetch the table from database
                    df_to_analyze = execute_query(conn, f"SELECT * FROM {table_to_analyze}")
                    if isinstance(df_to_analyze, pd.DataFrame):
                        table_exists = True
                
                if table_exists and df_to_analyze is not None:
                    response_content = f"Analyzing potential conflicts in table '{table_to_analyze}'...\n\n"
                    conflicts_df = analyze_conflicts(df_to_analyze)
                    
                    if len(conflicts_df) > 0:
                        response_content += f"Found {len(conflicts_df)} potential conflicts."
                        response_df = conflicts_df
                        
                        # Store conflicts in session state
                        st.session_state.conflicts_data[table_to_analyze] = conflicts_df
                        
                        # Check if user wants to save conflicts to database
                        save_conflicts = re.search(r'\b(save|store|simpan)\b', user_input.lower())
                        
                        if save_conflicts and conn and engine:
                            conflicts_table_name = f"{table_to_analyze}_conflicts"
                            success, message = save_conflicts_to_table(conflicts_df, conflicts_table_name, conn, engine)
                            response_content += f"\n\n{message}"
                    else:
                        response_content += "No conflicts found with current criteria."
                else:
                    response_content = f"Table '{table_to_analyze}' not found."
            else:
                response_content = "Please specify which table to analyze for conflicts."
        
        # Check for showing conflicts on map request
        elif re.search(r'\b(show|display|tampilkan|lihat)\s+(conflicts|interference|konflik|interferensi)\s+(?:on|in|di|pada)\s+(?:peta|map)\b', user_input.lower()):
            conflict_map_match = re.search(r'(conflicts|interference|konflik|interferensi)\s+(?:on|in|di|pada)\s+(?:peta|map)\s+(?:of|for|from|dari|untuk)\s+(\w+)', user_input.lower())
            table_to_map = None
            
            if conflict_map_match:
                table_to_map = conflict_map_match.group(2)
            elif len(st.session_state.dataframes) == 1:
                table_to_map = list(st.session_state.dataframes.keys())[0]
            
            if table_to_map:
                # Check if conflicts for this table exist in session state
                conflicts_exist = table_to_map in st.session_state.conflicts_data
                conflicts_df = st.session_state.conflicts_data.get(table_to_map)
                
                # Check if the table exists
                table_exists = False
                df_to_map = None
                
                if table_to_map in st.session_state.dataframes:
                    df_to_map = st.session_state.dataframes[table_to_map]
                    table_exists = True
                elif conn and table_to_map in db_tables:
                    # Fetch the table from database
                    df_to_map = execute_query(conn, f"SELECT * FROM {table_to_map}")
                    if isinstance(df_to_map, pd.DataFrame):
                        table_exists = True
                
                # Check if conflicts table exists in database
                conflicts_table_name = f"{table_to_map}_conflicts"
                conflicts_table_exists = False
                
                if conn and conflicts_table_name in db_tables:
                    conflicts_df = execute_query(conn, f"SELECT * FROM {conflicts_table_name}")
                    if isinstance(conflicts_df, pd.DataFrame):
                        conflicts_exist = True
                        conflicts_table_exists = True
                
                if table_exists and df_to_map is not None:
                    if conflicts_exist and conflicts_df is not None and not conflicts_df.empty:
                        response_content = f"Displaying map with {len(conflicts_df)} conflicts for table '{table_to_map}'..."
                        response_map = create_map_with_conflicts(df_to_map, conflicts_df, return_map=True)
                    else:
                        # No conflicts found, analyze them first
                        response_content = f"No saved conflicts found for '{table_to_map}'. Analyzing conflicts first...\n\n"
                        conflicts_df = analyze_conflicts(df_to_map)
                        
                        if len(conflicts_df) > 0:
                            response_content += f"Found {len(conflicts_df)} potential conflicts. Displaying on map..."
                            st.session_state.conflicts_data[table_to_map] = conflicts_df
                            response_map = create_map_with_conflicts(df_to_map, conflicts_df, return_map=True)
                        else:
                            response_content += "No conflicts found. Displaying regular map..."
                            response_map = create_map(df_to_map, return_map=True)
                else:
                    response_content = f"Table '{table_to_map}' not found."
            else:
                response_content = "Please specify which table to show conflicts for."
        
        # Check for saving conflicts request
        elif re.search(r'\b(save|store|simpan)\s+(conflicts|interference|konflik|interferensi)\b', user_input.lower()):
            save_conflicts_match = re.search(r'(conflicts|interference|konflik|interferensi)\s+(?:for|from|dari|untuk)\s+(\w+)', user_input.lower())
            table_to_save = None
            
            if save_conflicts_match:
                table_to_save = save_conflicts_match.group(2)
            elif len(st.session_state.dataframes) == 1:
                table_to_save = list(st.session_state.dataframes.keys())[0]
            
            if table_to_save:
                # Check if conflicts for this table exist in session state
                conflicts_exist = table_to_save in st.session_state.conflicts_data
                conflicts_df = st.session_state.conflicts_data.get(table_to_save)
                
                if conflicts_exist and conflicts_df is not None and not conflicts_df.empty:
                    if conn and engine:
                        conflicts_table_name = f"{table_to_save}_conflicts"
                        success, message = save_conflicts_to_table(conflicts_df, conflicts_table_name, conn, engine)
                        response_content = message
                    else:
                        response_content = "Database connection not available. Could not save conflicts."
                else:
                    # No conflicts found, need to analyze them first
                    response_content = f"No conflicts analyzed yet for '{table_to_save}'. Please run conflict analysis first."
            else:
                response_content = "Please specify which table's conflicts to save."
        
        # Prepare prompt for Gemini if no special commands were processed
        if not response_content:
            prompt = f"""
            I have a PostgreSQL database with the following tables:
            {tables_info}
            
            User query: {processed_query}
            
            If the query asks for data from the database, please generate a valid PostgreSQL SQL query.
            If it's a general question about the data structure, provide helpful information.
            Format your response as follows:
            
            [SQL]
            SQL query here if applicable
            [/SQL]
            
            Then provide a human-friendly explanation.
            """
            
            # Get response from Gemini
            response = model.generate_content(prompt)
            
            # Process the response to extract SQL query if present
            content = response.text
            sql_match = re.search(r'\[SQL\](.*?)\[\/SQL\]', content, re.DOTALL)
            
            if sql_match:
                sql_query = sql_match.group(1).strip()
                if sql_query and not sql_query.lower().startswith('no sql'):
                    try:
                        # Execute the SQL query
                        if conn:
                            result = execute_query(conn, sql_query)
                            if isinstance(result, pd.DataFrame):
                                response_content = f"Here's the result from your query:\n\n"
                                response_content += content.replace(sql_match.group(0), "")
                                response_df = result
                            else:
                                response_content = f"{result}\n\n"
                                response_content += content.replace(sql_match.group(0), "")
                        else:
                            # Try to execute against session dataframes if possible
                            # This would require parsing the SQL and executing against pandas
                            # For simplicity, just show that DB connection is not available
                            response_content = "Database connection not available. Could not execute SQL query.\n\n"
                            response_content += content.replace(sql_match.group(0), "")
                    except Exception as e:
                        response_content = f"Error executing SQL query: {e}\n\n"
                        response_content += content.replace(sql_match.group(0), "")
                else:
                    response_content = content.replace(sql_match.group(0), "")
            else:
                response_content = content
        
        # Add assistant message to chat history with any dataframe or map
        response_message = {"role": "assistant", "content": response_content}
        if response_df is not None:
            response_message["dataframe"] = response_df
        if response_map is not None:
            response_message["map"] = response_map
        
        st.session_state.chat_history.append(response_message)
        
        # Display assistant message
        with st.chat_message("assistant"):
            st.write(response_content)
            if response_df is not None:
                st.dataframe(response_df)
            if response_map is not None:
                if isinstance(response_map, str):
                    st.info(response_map)
                else:
                    try:
                        # Try to display the map in Streamlit
                        response_map.to_streamlit(height=600)
                    except Exception as e:
                        st.error(f"Could not display map: {e}")

if __name__ == "__main__":
    main()
