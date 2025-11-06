"""
PHASE 1: AIS 데이터 필터링 및 전처리
- Step 0: ship_id 디코딩 (Base64)
- Step 1: vessel_info와 매칭
- Step 2: 선박 종류 필터링 (화물선 70-79, 탱커선 80-89)
- Step 3-0: 유효하지 않은 좌표 제거

출력: filtered_ais_data.csv
"""

import pandas as pd
import base64
import sys
import os

# --- File Paths ---
BASE_PATH = r"D:\2025-MSP\test"
VESSEL_INFO_PATH = os.path.join(BASE_PATH, "2024_tm_vessel_info.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "filtered_ais_data.csv")

# --- Constants ---
CHUNK_SIZE = 1000000  # Process 1 million rows at a time

def decode_ship_id(encoded_id):
    """Decodes a Base64 encoded ship_id."""
    try:
        if pd.isna(encoded_id):
            return None
        if isinstance(encoded_id, str):
            encoded_id = encoded_id.encode('utf-8')
        decoded_bytes = base64.b64decode(encoded_id)
        return decoded_bytes.decode('utf-8')
    except Exception:
        return None

def is_valid_coordinate(lon, lat):
    """Check if coordinates are valid."""
    try:
        return (-180 <= lon <= 180) and (-90 <= lat <= 90)
    except:
        return False

def process_chunk(ais_chunk_df, vessel_info_df):
    """
    Process AIS chunk:
    - Match with vessel info (both are Base64 encoded, no need to decode)
    - Filter by vessel type
    - Remove invalid coordinates
    - Decode ship_id at the end for readability
    """
    # Step 1: Match with static info (merge) - both ship_ids are Base64 encoded
    merged_df = pd.merge(
        ais_chunk_df, 
        vessel_info_df, 
        on='ship_id',  # Direct match on encoded ship_id
        how='inner',
        suffixes=('', '_vessel')
    )
    
    if merged_df.empty:
        return pd.DataFrame()

    # Step 2: Filter by vessel type ('ship_kind')
    cargo_mask = (merged_df['ship_kind'] >= 70) & (merged_df['ship_kind'] <= 79)
    tanker_mask = (merged_df['ship_kind'] >= 80) & (merged_df['ship_kind'] <= 89)
    filtered_df = merged_df[cargo_mask | tanker_mask].copy()
    
    if filtered_df.empty:
        return pd.DataFrame()

    # Step 3-0: Filter invalid coordinates
    valid_coords_mask = filtered_df.apply(
        lambda row: is_valid_coordinate(row['lon_val'], row['lat_val']), 
        axis=1
    )
    filtered_df = filtered_df[valid_coords_mask].copy()
    
    # Step 0: Decode ship_id for readability (after all filtering)
    filtered_df['decoded_ship_id'] = filtered_df['ship_id'].apply(decode_ship_id)
    
    # Keep only necessary columns
    keep_columns = ['ship_id', 'decoded_ship_id', 'recv_dt', 'lon_val', 'lat_val', 
                   'sog_val', 'cog_val', 'hdg_val', 'ship_kind']
    
    # Check which columns exist
    available_columns = [col for col in keep_columns if col in filtered_df.columns]
    filtered_df = filtered_df[available_columns]

    return filtered_df

def main():
    """
    Main processing pipeline for Phase 1.
    """
    print("=" * 70)
    print("PHASE 1: AIS Data Filtering and Preprocessing")
    print("=" * 70)
    
    # Load vessel info
    print("\nLoading vessel information...")
    try:
        vessel_info_df = pd.read_csv(VESSEL_INFO_PATH, dtype={'ship_id': str})
        
        # Convert ship_kind to numeric, coerce non-numeric to NaN
        vessel_info_df['ship_kind'] = pd.to_numeric(vessel_info_df['ship_kind'], errors='coerce')
        
        # Fill NaN with 0 or drop them
        vessel_info_df['ship_kind'] = vessel_info_df['ship_kind'].fillna(0).astype(int)
        
        print(f"Loaded {len(vessel_info_df)} vessel records.")
    except FileNotFoundError:
        print(f"ERROR: Vessel info file not found at {VESSEL_INFO_PATH}")
        return
    except Exception as e:
        print(f"ERROR loading vessel info file: {e}")
        return

    # Generate list of AIS files for December
    ais_file_paths = [
        os.path.join(BASE_PATH, f"th_ais_202412{day:02d}.csv") 
        for day in range(1, 32)
    ]

    first_chunk_written = False
    total_input_rows = 0
    total_output_rows = 0
    files_processed = 0

    # Loop through all AIS files
    for file_idx, ais_path in enumerate(ais_file_paths, 1):
        if not os.path.exists(ais_path):
            print(f"\nWarning: File not found, skipping: {ais_path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{file_idx}/31] Processing: {os.path.basename(ais_path)}")
        print(f"{'=' * 70}")
        
        try:
            chunk_iter = pd.read_csv(ais_path, chunksize=CHUNK_SIZE, low_memory=False)
        except Exception as e:
            print(f"ERROR reading file. Skipping. Error: {e}")
            continue
        
        file_input_rows = 0
        file_output_rows = 0
        
        for chunk_idx, chunk in enumerate(chunk_iter, 1):
            chunk_size = len(chunk)
            file_input_rows += chunk_size
            total_input_rows += chunk_size
            
            print(f"  Chunk {chunk_idx}: {chunk_size:,} rows", end=" → ")
            
            processed_chunk = process_chunk(chunk, vessel_info_df)
            
            if not processed_chunk.empty:
                rows_found = len(processed_chunk)
                file_output_rows += rows_found
                total_output_rows += rows_found
                print(f"{rows_found:,} rows kept")
                
                # Write to CSV
                if not first_chunk_written:
                    processed_chunk.to_csv(OUTPUT_PATH, index=False, mode='w')
                    first_chunk_written = True
                else:
                    processed_chunk.to_csv(OUTPUT_PATH, index=False, mode='a', header=False)
            else:
                print("0 rows kept")
        
        files_processed += 1
        print(f"  File summary: {file_input_rows:,} → {file_output_rows:,} rows "
              f"({file_output_rows/file_input_rows*100:.1f}%)")

    # Final summary
    print(f"\n{'=' * 70}")
    print("PHASE 1 COMPLETE")
    print(f"{'=' * 70}")
    
    if not first_chunk_written:
        print("⚠️  No relevant data found in any file.")
        print("    Check vessel_info file and ship_kind values.")
    else:
        print(f"✓ Files processed: {files_processed}/31")
        print(f"✓ Total input rows: {total_input_rows:,}")
        print(f"✓ Total output rows: {total_output_rows:,}")
        print(f"✓ Retention rate: {total_output_rows/total_input_rows*100:.2f}%")
        print(f"✓ Output file: {OUTPUT_PATH}")
        print(f"\nNext step: Run phase2_trajectory_analysis.py")

if __name__ == "__main__":
    main()
