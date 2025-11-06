"""
PHASE 1 (병렬 버전): AIS 데이터 필터링 및 날짜별 저장
- 각 날짜별로 개별 filtered 파일 생성
- Phase 2에서 병렬 처리 가능하도록
"""

import pandas as pd
import base64
import sys
import os

# --- File Paths ---
BASE_PATH = r"/media/data1/cmyoon/AIS_process"
VESSEL_INFO_PATH = os.path.join(BASE_PATH, "2024_tm_vessel_info.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "filtered_daily")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    - Match with vessel info (both Base64 encoded)
    - Filter by vessel type
    - Remove invalid coordinates
    - Decode ship_id for readability
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
    Main processing pipeline for Phase 1 (parallel version).
    """
    print("=" * 70)
    print("PHASE 1: AIS Data Filtering (Daily Split for Parallel Processing)")
    print("=" * 70)
    
    # Load vessel info
    print("\nLoading vessel information...")
    try:
        vessel_info_df = pd.read_csv(VESSEL_INFO_PATH, dtype={'ship_id': str})
        
        # Convert ship_kind to numeric
        vessel_info_df['ship_kind'] = pd.to_numeric(vessel_info_df['ship_kind'], errors='coerce')
        vessel_info_df['ship_kind'] = vessel_info_df['ship_kind'].fillna(0).astype(int)
        
        print(f"✓ Loaded {len(vessel_info_df):,} vessel records.")
        
        # Count target vessels
        cargo_count = ((vessel_info_df['ship_kind'] >= 70) & (vessel_info_df['ship_kind'] <= 79)).sum()
        tanker_count = ((vessel_info_df['ship_kind'] >= 80) & (vessel_info_df['ship_kind'] <= 89)).sum()
        print(f"  - Cargo ships (70-79): {cargo_count:,}")
        print(f"  - Tanker ships (80-89): {tanker_count:,}")
        print(f"  - Total target ships: {cargo_count + tanker_count:,}")
        
    except FileNotFoundError:
        print(f"ERROR: Vessel info file not found at {VESSEL_INFO_PATH}")
        return
    except Exception as e:
        print(f"ERROR loading vessel info file: {e}")
        return

    # Generate list of AIS files for December
    ais_file_paths = [
        (day, os.path.join(BASE_PATH, f"th_ais_202412{day:02d}.csv"))
        for day in range(1, 32)
    ]

    total_input_rows = 0
    total_output_rows = 0
    files_processed = 0
    daily_stats = []

    # Loop through all AIS files
    for day, ais_path in ais_file_paths:
        if not os.path.exists(ais_path):
            print(f"\nWarning: File not found, skipping: {ais_path}")
            continue

        output_path = os.path.join(OUTPUT_DIR, f"filtered_202412{day:02d}.csv")
        
        print(f"\n{'=' * 70}")
        print(f"[Day {day}/31] Processing: {os.path.basename(ais_path)}")
        print(f"{'=' * 70}")
        
        try:
            chunk_iter = pd.read_csv(ais_path, chunksize=CHUNK_SIZE, low_memory=False)
        except Exception as e:
            print(f"ERROR reading file. Skipping. Error: {e}")
            continue
        
        first_chunk_written = False
        file_input_rows = 0
        file_output_rows = 0
        
        for chunk_idx, chunk in enumerate(chunk_iter, 1):
            chunk_size = len(chunk)
            file_input_rows += chunk_size
            
            print(f"  Chunk {chunk_idx}: {chunk_size:,} rows", end=" → ")
            
            processed_chunk = process_chunk(chunk, vessel_info_df)
            
            if not processed_chunk.empty:
                rows_found = len(processed_chunk)
                file_output_rows += rows_found
                print(f"{rows_found:,} rows kept")
                
                # Write to daily CSV
                if not first_chunk_written:
                    processed_chunk.to_csv(output_path, index=False, mode='w')
                    first_chunk_written = True
                else:
                    processed_chunk.to_csv(output_path, index=False, mode='a', header=False)
            else:
                print("0 rows kept")
        
        if first_chunk_written:
            files_processed += 1
            total_input_rows += file_input_rows
            total_output_rows += file_output_rows
            
            retention_rate = file_output_rows / file_input_rows * 100 if file_input_rows > 0 else 0
            print(f"  ✓ Day {day} summary: {file_input_rows:,} → {file_output_rows:,} rows ({retention_rate:.1f}%)")
            print(f"  ✓ Saved to: {output_path}")
            
            daily_stats.append({
                'day': day,
                'input_rows': file_input_rows,
                'output_rows': file_output_rows,
                'retention_rate': retention_rate
            })
        else:
            print(f"  ⚠️  No data found for day {day}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("PHASE 1 COMPLETE")
    print(f"{'=' * 70}")
    
    if files_processed == 0:
        print("⚠️  No relevant data found in any file.")
    else:
        print(f"✓ Files processed: {files_processed}/31")
        print(f"✓ Total input rows: {total_input_rows:,}")
        print(f"✓ Total output rows: {total_output_rows:,}")
        print(f"✓ Overall retention rate: {total_output_rows/total_input_rows*100:.2f}%")
        print(f"✓ Output directory: {OUTPUT_DIR}")
        
        # Show daily breakdown
        print(f"\nDaily breakdown:")
        for stat in daily_stats[:5]:  # Show first 5
            print(f"  Day {stat['day']:2d}: {stat['output_rows']:>10,} rows ({stat['retention_rate']:>5.1f}%)")
        if len(daily_stats) > 5:
            print(f"  ... ({len(daily_stats)-5} more days)")
        
        print(f"\n{'=' * 70}")
        print("Next step: Run phase2_parallel_trajectory.py")
        print("  - This will process each day in parallel")
        print("  - Much faster than single-threaded processing")
        print(f"{'=' * 70}")

if __name__ == "__main__":
    main()