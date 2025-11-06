import pandas as pd
import base64
import sys
import os

# --- File Paths ---
BASE_PATH = r"D:\2025-MSP\test"
VESSEL_INFO_PATH = os.path.join(BASE_PATH, "2024_tm_vessel_info.csv")
GRID_PATH = os.path.join(BASE_PATH, "grid_polygon_wkt.csv")
INTERMEDIATE_OUTPUT_PATH = os.path.join(BASE_PATH, "filtered_ais_data.csv")

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
        return None # Return None if decoding fails

def process_chunk(ais_chunk_df, vessel_info_df):
    """
    Decodes ship_id, merges with vessel info, and filters by vessel type.
    Covers steps 0, 1, 2.
    """
    # Step 0: Decode ship_id
    ais_chunk_df['decoded_ship_id'] = ais_chunk_df['ship_id'].apply(decode_ship_id)
    
    # Drop rows where decoding failed
    ais_chunk_df.dropna(subset=['decoded_ship_id'], inplace=True)

    # Step 1: Match with static info (merge)
    merged_df = pd.merge(ais_chunk_df, vessel_info_df, left_on='decoded_ship_id', right_on='ship_id', how='inner')

    # Step 2: Filter by vessel type ('ship_kind')
    cargo_mask = (merged_df['ship_kind'] >= 70) & (merged_df['ship_kind'] <= 79)
    tanker_mask = (merged_df['ship_kind'] >= 80) & (merged_df['ship_kind'] <= 89)
    filtered_df = merged_df[cargo_mask | tanker_mask].copy()

    return filtered_df

def main():
    """
    Main processing pipeline.
    """
    print("Loading auxiliary data...")
    try:
        # Specify dtype for ship_id to avoid potential type mismatches during merge
        vessel_info_df = pd.read_csv(VESSEL_INFO_PATH, dtype={'ship_id': str})
    except FileNotFoundError:
        print(f"Error: Auxiliary file not found at {VESSEL_INFO_PATH}")
        return
    except Exception as e:
        print(f"Error loading vessel info file: {e}")
        return

    # --- Generate list of AIS files for December ---
    ais_file_paths = [os.path.join(BASE_PATH, f"th_ais_202412{day:02d}.csv") for day in range(1, 32)]

    first_chunk_written = False
    total_relevant_rows = 0

    # --- Loop through all AIS files ---
    for ais_path in ais_file_paths:
        if not os.path.exists(ais_path):
            print(f"Warning: File not found, skipping: {ais_path}")
            continue

        print(f"\n===== Processing File: {os.path.basename(ais_path)} =====")
        
        try:
            chunk_iter = pd.read_csv(ais_path, chunksize=CHUNK_SIZE, low_memory=False)
        except Exception as e:
            print(f"Error reading {ais_path}. Skipping. Error: {e}")
            continue
        
        for i, chunk in enumerate(chunk_iter):
            print(f"--- Processing chunk {i+1} from {os.path.basename(ais_path)} ---")
            
            processed_chunk = process_chunk(chunk, vessel_info_df)
            
            if not processed_chunk.empty:
                rows_found = len(processed_chunk)
                total_relevant_rows += rows_found
                print(f"Found {rows_found} relevant rows.")
                
                if not first_chunk_written:
                    processed_chunk.to_csv(INTERMEDIATE_OUTPUT_PATH, index=False, mode='w')
                    first_chunk_written = True
                else:
                    processed_chunk.to_csv(INTERMEDIATE_OUTPUT_PATH, index=False, mode='a', header=False)
            else:
                print("No relevant rows found in this chunk.")

    if not first_chunk_written:
        print("\nProcessing complete. No relevant data was found in any file.")
    else:
        print(f"\nProcessing complete for all files. Total relevant rows: {total_relevant_rows}")
        print(f"Intermediate filtered data saved to: {INTERMEDIATE_OUTPUT_PATH}")
    
    print("\nNext steps: Create trajectories, cut by grid, and calculate lengths.")

if __name__ == "__main__":
    main()
