"""
Phase 1 디버깅 스크립트
문제가 어디서 발생하는지 확인합니다.
"""

import pandas as pd
import base64
import os

# --- File Paths ---
BASE_PATH = "/media/data1/cmyoon/AIS_process"
VESSEL_INFO_PATH = os.path.join(BASE_PATH, "2024_tm_vessel_info.csv")
AIS_PATH = os.path.join(BASE_PATH, "th_ais_20241201.csv")  # 첫날 파일만 테스트

def decode_ship_id(encoded_id):
    """Decodes a Base64 encoded ship_id."""
    try:
        if pd.isna(encoded_id):
            return None
        if isinstance(encoded_id, str):
            encoded_id = encoded_id.encode('utf-8')
        decoded_bytes = base64.b64decode(encoded_id)
        return decoded_bytes.decode('utf-8')
    except Exception as e:
        return None

print("=" * 70)
print("PHASE 1 DEBUGGING")
print("=" * 70)

# ============================================
# 1. Vessel Info 파일 확인
# ============================================
print("\n[1] Checking vessel_info file...")
vessel_info_df = pd.read_csv(VESSEL_INFO_PATH, dtype={'ship_id': str})

print(f"✓ Loaded {len(vessel_info_df):,} rows")
print(f"\nColumns: {vessel_info_df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(vessel_info_df.head())

print(f"\nship_id sample values:")
print(vessel_info_df['ship_id'].head(10).tolist())

if 'ship_kind' in vessel_info_df.columns:
    print(f"\nship_kind original type: {vessel_info_df['ship_kind'].dtype}")
    print(f"ship_kind sample values (before conversion):")
    print(vessel_info_df['ship_kind'].head(20).tolist())
    
    # Convert to numeric
    print(f"\nConverting ship_kind to numeric...")
    vessel_info_df['ship_kind'] = pd.to_numeric(vessel_info_df['ship_kind'], errors='coerce')
    vessel_info_df['ship_kind'] = vessel_info_df['ship_kind'].fillna(0).astype(int)
    
    print(f"ship_kind after conversion type: {vessel_info_df['ship_kind'].dtype}")
    print(f"\nship_kind statistics:")
    print(vessel_info_df['ship_kind'].describe())
    print(f"\nship_kind value counts (top 20):")
    print(vessel_info_df['ship_kind'].value_counts().head(20))
    
    # Check target ranges
    cargo_count = ((vessel_info_df['ship_kind'] >= 70) & (vessel_info_df['ship_kind'] <= 79)).sum()
    tanker_count = ((vessel_info_df['ship_kind'] >= 80) & (vessel_info_df['ship_kind'] <= 89)).sum()
    print(f"\nCargo ships (70-79): {cargo_count:,}")
    print(f"Tanker ships (80-89): {tanker_count:,}")
    print(f"Total target ships: {cargo_count + tanker_count:,}")
else:
    print("\n⚠️ WARNING: 'ship_kind' column not found!")

# ============================================
# 2. AIS 파일 확인
# ============================================
print("\n" + "=" * 70)
print("[2] Checking AIS file (first 10,000 rows)...")
ais_df = pd.read_csv(AIS_PATH, nrows=10000)

print(f"✓ Loaded {len(ais_df):,} rows")
print(f"\nColumns: {ais_df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(ais_df.head())

print(f"\nship_id sample values (encoded):")
print(ais_df['ship_id'].head(10).tolist())

# ============================================
# 3. ship_id 디코딩 테스트
# ============================================
print("\n" + "=" * 70)
print("[3] Testing ship_id decoding...")
ais_df['decoded_ship_id'] = ais_df['ship_id'].apply(decode_ship_id)

successful_decodes = ais_df['decoded_ship_id'].notna().sum()
print(f"✓ Successfully decoded: {successful_decodes:,} / {len(ais_df):,} ({successful_decodes/len(ais_df)*100:.1f}%)")

if successful_decodes > 0:
    print(f"\nDecoded ship_id samples:")
    print(ais_df[ais_df['decoded_ship_id'].notna()]['decoded_ship_id'].head(10).tolist())
    
    # Remove failed decodes
    ais_df_decoded = ais_df.dropna(subset=['decoded_ship_id'])
else:
    print("⚠️ WARNING: No ship_ids were successfully decoded!")
    print("\nTrying alternative decoding method...")
    
    # Try without encoding to utf-8 first
    def decode_ship_id_v2(encoded_id):
        try:
            if pd.isna(encoded_id):
                return None
            decoded_bytes = base64.b64decode(encoded_id)
            return decoded_bytes.decode('utf-8')
        except Exception as e:
            return None
    
    ais_df['decoded_ship_id'] = ais_df['ship_id'].apply(decode_ship_id_v2)
    successful_decodes = ais_df['decoded_ship_id'].notna().sum()
    print(f"Alternative method: {successful_decodes:,} / {len(ais_df):,}")
    
    if successful_decodes == 0:
        print("\n❌ CRITICAL: Cannot decode any ship_ids!")
        print("Sample encoded values to check manually:")
        for val in ais_df['ship_id'].head(5):
            print(f"  '{val}'")
        exit(1)
    
    ais_df_decoded = ais_df.dropna(subset=['decoded_ship_id'])

# ============================================
# 4. vessel_info 매칭 테스트
# ============================================
print("\n" + "=" * 70)
print("[4] Testing vessel_info matching...")

print(f"\nAIS decoded ship_id type: {ais_df_decoded['decoded_ship_id'].dtype}")
print(f"Vessel info ship_id type: {vessel_info_df['ship_id'].dtype}")

# Try matching
merged_df = pd.merge(
    ais_df_decoded, 
    vessel_info_df, 
    left_on='decoded_ship_id', 
    right_on='ship_id', 
    how='inner',
    suffixes=('', '_vessel')
)

print(f"✓ Matched rows: {len(merged_df):,} / {len(ais_df_decoded):,} ({len(merged_df)/len(ais_df_decoded)*100:.1f}%)")

if len(merged_df) == 0:
    print("\n⚠️ WARNING: No matches found!")
    print("\nComparing sample IDs:")
    print(f"AIS decoded_ship_id samples: {ais_df_decoded['decoded_ship_id'].head(5).tolist()}")
    print(f"Vessel info ship_id samples: {vessel_info_df['ship_id'].head(5).tolist()}")
    
    # Check if any overlap exists
    ais_ids = set(ais_df_decoded['decoded_ship_id'].unique())
    vessel_ids = set(vessel_info_df['ship_id'].unique())
    overlap = ais_ids.intersection(vessel_ids)
    print(f"\nOverlapping ship_ids: {len(overlap):,}")
    
    if len(overlap) > 0:
        print(f"Sample overlapping IDs: {list(overlap)[:5]}")
    else:
        print("❌ CRITICAL: No overlapping ship_ids between AIS and vessel_info!")
    exit(1)

# ============================================
# 5. ship_kind 필터링 테스트
# ============================================
print("\n" + "=" * 70)
print("[5] Testing ship_kind filtering...")

if 'ship_kind' not in merged_df.columns:
    print("❌ CRITICAL: 'ship_kind' column not found after merge!")
    print(f"Available columns: {merged_df.columns.tolist()}")
    exit(1)

print(f"\nship_kind distribution in merged data:")
print(merged_df['ship_kind'].value_counts().head(20))

cargo_mask = (merged_df['ship_kind'] >= 70) & (merged_df['ship_kind'] <= 79)
tanker_mask = (merged_df['ship_kind'] >= 80) & (merged_df['ship_kind'] <= 89)
filtered_df = merged_df[cargo_mask | tanker_mask]

print(f"\n✓ Filtered rows: {len(filtered_df):,} / {len(merged_df):,} ({len(filtered_df)/len(merged_df)*100:.1f}%)")

if len(filtered_df) == 0:
    print("\n⚠️ WARNING: No cargo/tanker ships found in this sample!")
    print("This might be normal if the sample is small.")
else:
    print(f"\nFiltered ship_kind distribution:")
    print(filtered_df['ship_kind'].value_counts())

# ============================================
# 6. 좌표 유효성 검사
# ============================================
print("\n" + "=" * 70)
print("[6] Testing coordinate validity...")

if len(filtered_df) > 0:
    valid_coords = filtered_df[
        (filtered_df['lon_val'] >= -180) & (filtered_df['lon_val'] <= 180) &
        (filtered_df['lat_val'] >= -90) & (filtered_df['lat_val'] <= 90)
    ]
    
    print(f"✓ Valid coordinates: {len(valid_coords):,} / {len(filtered_df):,} ({len(valid_coords)/len(filtered_df)*100:.1f}%)")
    
    if len(valid_coords) > 0:
        print("\n✅ SUCCESS: Data pipeline should work!")
        print(f"\nFinal output would be: {len(valid_coords):,} rows from this sample")
    else:
        print("\n⚠️ All coordinates are invalid in filtered data!")
else:
    print("Cannot test - no filtered data available")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print(f"1. Vessel info loaded: ✓")
print(f"2. AIS data loaded: ✓")
print(f"3. ship_id decoding: {'✓' if successful_decodes > 0 else '✗'}")
print(f"4. vessel_info matching: {'✓' if len(merged_df) > 0 else '✗'}")
print(f"5. ship_kind filtering: {'✓' if len(filtered_df) > 0 else '⚠️  (might be normal)'}")
print(f"6. Coordinate validity: {'✓' if len(filtered_df) > 0 and len(valid_coords) > 0 else 'N/A'}")