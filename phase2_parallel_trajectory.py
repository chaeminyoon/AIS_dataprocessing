"""
PHASE 2 (병렬 버전): 궤적 생성 및 격자 분석
- 날짜별로 병렬 처리
- 멀티프로세싱으로 속도 향상
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely import wkt
import json
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
import glob

# --- File Paths ---
BASE_PATH = r"/media/data1/cmyoon/AIS_process"
INPUT_DIR = os.path.join(BASE_PATH, "filtered_daily")
GRID_PATH = os.path.join(BASE_PATH, "grid_polygon_wkt.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "results")
FINAL_CSV = os.path.join(OUTPUT_DIR, "grid_trajectory_summary.csv")
FINAL_JSON = os.path.join(OUTPUT_DIR, "grid_trajectory_summary.json")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of parallel processes (use all CPUs - 1)
NUM_PROCESSES = max(1, cpu_count() - 1)

def create_trajectories(df):
    """Create trajectories for each ship_id."""
    # Sort by ship_id and time
    df_sorted = df.sort_values(['decoded_ship_id', 'recv_dt']).copy()
    
    trajectories = []
    grouped = df_sorted.groupby('decoded_ship_id')
    
    for ship_id, group in grouped:
        if len(group) < 2:
            continue
        
        try:
            points = [Point(row['lon_val'], row['lat_val']) for _, row in group.iterrows()]
            line = LineString(points)
            trajectories.append({
                'ship_id': ship_id,
                'geometry': line,
                'point_count': len(points)
            })
        except Exception:
            continue
    
    gdf_trajectories = gpd.GeoDataFrame(trajectories, crs='EPSG:4326')
    return gdf_trajectories

def intersect_with_grid(gdf_trajectories, gdf_grid):
    """Intersect trajectories with grid cells."""
    results = []
    spatial_index = gdf_grid.sindex
    
    for idx, traj_row in gdf_trajectories.iterrows():
        ship_id = traj_row['ship_id']
        trajectory = traj_row['geometry']
        
        # Get candidate grids using spatial index
        possible_matches_idx = list(spatial_index.intersection(trajectory.bounds))
        possible_matches = gdf_grid.iloc[possible_matches_idx]
        
        # Precise intersection check
        intersecting_grids = possible_matches[possible_matches.intersects(trajectory)]
        
        for _, grid_row in intersecting_grids.iterrows():
            grid_id = grid_row['MIN1']
            grid_geom = grid_row['geometry']
            
            try:
                intersection = trajectory.intersection(grid_geom)
                
                if intersection.is_empty:
                    continue
                
                if intersection.geom_type == 'LineString':
                    results.append({
                        'grid_id': grid_id,
                        'ship_id': ship_id,
                        'geometry': intersection
                    })
                elif intersection.geom_type == 'MultiLineString':
                    for line in intersection.geoms:
                        results.append({
                            'grid_id': grid_id,
                            'ship_id': ship_id,
                            'geometry': line
                        })
                elif intersection.geom_type == 'GeometryCollection':
                    for geom in intersection.geoms:
                        if geom.geom_type == 'LineString':
                            results.append({
                                'grid_id': grid_id,
                                'ship_id': ship_id,
                                'geometry': geom
                            })
            except Exception:
                continue
    
    if not results:
        return gpd.GeoDataFrame()
    
    gdf_intersections = gpd.GeoDataFrame(results, crs='EPSG:4326')
    return gdf_intersections

def process_single_day(args):
    """Process a single day's data - for parallel execution."""
    day_file, gdf_grid = args
    day_num = os.path.basename(day_file).replace('filtered_', '').replace('.csv', '')
    
    print(f"[Day {day_num}] Starting...")
    
    try:
        # Load data
        df = pd.read_csv(day_file, low_memory=False)
        if df.empty:
            print(f"[Day {day_num}] No data, skipping")
            return None
        
        print(f"[Day {day_num}] Loaded {len(df):,} rows")
        
        # Create trajectories
        gdf_traj = create_trajectories(df)
        if gdf_traj.empty:
            print(f"[Day {day_num}] No trajectories created")
            return None
        
        print(f"[Day {day_num}] Created {len(gdf_traj):,} trajectories")
        
        # Intersect with grid
        gdf_int = intersect_with_grid(gdf_traj, gdf_grid)
        if gdf_int.empty:
            print(f"[Day {day_num}] No intersections found")
            return None
        
        print(f"[Day {day_num}] Found {len(gdf_int):,} intersections")
        
        # Calculate lengths
        gdf_int_proj = gdf_int.to_crs('EPSG:3857')
        gdf_int_proj['length_m'] = gdf_int_proj.geometry.length
        
        # Aggregate by grid
        summary = gdf_int_proj.groupby('grid_id').agg(
            length_sum=('length_m', 'sum'),
            count=('length_m', 'count')
        ).reset_index()
        
        print(f"[Day {day_num}] ✓ Complete: {len(summary):,} grids")
        
        return summary
        
    except Exception as e:
        print(f"[Day {day_num}] ERROR: {e}")
        return None

def main():
    """Main processing pipeline for Phase 2 (parallel version)."""
    start_time = datetime.now()
    
    print("=" * 70)
    print("PHASE 2: Trajectory Analysis (Parallel Processing)")
    print("=" * 70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU cores available: {cpu_count()}")
    print(f"Using {NUM_PROCESSES} parallel processes")
    
    # Check input directory
    if not os.path.exists(INPUT_DIR):
        print(f"\n❌ ERROR: Input directory not found: {INPUT_DIR}")
        print("   Please run phase1_parallel_filtering.py first!")
        return
    
    # Get list of daily files
    daily_files = sorted(glob.glob(os.path.join(INPUT_DIR, "filtered_*.csv")))
    
    if not daily_files:
        print(f"\n❌ ERROR: No filtered files found in {INPUT_DIR}")
        return
    
    print(f"\n✓ Found {len(daily_files)} daily files to process")
    
    # Load grid
    print(f"\nLoading grid data...")
    try:
        grid_df = pd.read_csv(GRID_PATH)
        
        # Fix WKT format: replace semicolons with commas
        print(f"  Fixing WKT format (semicolon → comma)...")
        grid_df['wkt'] = grid_df['wkt'].str.replace(';', ',')
        
        # Parse WKT geometry
        grid_df['geometry'] = grid_df['wkt'].apply(wkt.loads)
        gdf_grid = gpd.GeoDataFrame(grid_df, geometry='geometry', crs='EPSG:4326')
        gdf_grid.sindex  # Create spatial index
        print(f"✓ Loaded {len(gdf_grid):,} grid cells")
    except Exception as e:
        print(f"❌ ERROR loading grid: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Prepare arguments for parallel processing
    args_list = [(day_file, gdf_grid) for day_file in daily_files]
    
    # Process in parallel
    print(f"\n{'=' * 70}")
    print(f"Processing {len(daily_files)} days in parallel...")
    print(f"{'=' * 70}\n")
    
    with Pool(NUM_PROCESSES) as pool:
        results = pool.map(process_single_day, args_list)
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("\n❌ ERROR: No valid results from any day")
        return
    
    print(f"\n{'=' * 70}")
    print(f"Combining results from {len(valid_results)} days...")
    print(f"{'=' * 70}")
    
    # Combine all daily results
    combined_df = pd.concat(valid_results, ignore_index=True)
    
    # Aggregate across all days
    final_summary = combined_df.groupby('grid_id').agg(
        선박선길이sum=('length_sum', 'sum'),
        선의갯수=('count', 'sum')
    ).reset_index()
    
    # Rename grid_id to 격자명
    final_summary.rename(columns={'grid_id': '격자명'}, inplace=True)
    
    # Round values
    final_summary['선박선길이sum'] = final_summary['선박선길이sum'].round(2)
    final_summary['선의갯수'] = final_summary['선의갯수'].astype(int)
    
    # Sort by grid name
    final_summary = final_summary.sort_values('격자명').reset_index(drop=True)
    
    # Save results
    print(f"\nSaving results...")
    final_summary.to_csv(FINAL_CSV, index=False, encoding='utf-8-sig')
    print(f"✓ CSV saved: {FINAL_CSV}")
    
    summary_dict = final_summary.to_dict(orient='records')
    with open(FINAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON saved: {FINAL_JSON}")
    
    # Display sample
    print(f"\nSample results (first 10 rows):")
    print(final_summary.head(10).to_string(index=False))
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE!")
    print("=" * 70)
    print(f"Start time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:    {duration}")
    print(f"\nResults:")
    print(f"  - Total grid cells: {len(final_summary):,}")
    print(f"  - Total trajectory length: {final_summary['선박선길이sum'].sum():,.2f} meters")
    print(f"  - Total trajectory segments: {final_summary['선의갯수'].sum():,}")
    print(f"\nOutput files:")
    print(f"  - CSV:  {FINAL_CSV}")
    print(f"  - JSON: {FINAL_JSON}")

if __name__ == "__main__":
    main()