"""
PHASE 2: 궤적 생성 및 격자 분석
- Step 3: ship_id별 궤적 생성 (LineString)
- Step 4: 격자와 교차 계산
- Step 5: 격자별 집계 (선박선길이sum, 선의갯수)

입력: filtered_ais_data.csv (Phase 1 출력)
출력: grid_trajectory_summary.csv, grid_trajectory_summary.json
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely import wkt
import json
import os
from datetime import datetime

# --- File Paths ---
BASE_PATH = "/media/data1/cmyoon/AIS_process"
INPUT_PATH = os.path.join(BASE_PATH, "filtered_ais_data.csv")
GRID_PATH = os.path.join(BASE_PATH, "grid_polygon_wkt.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "grid_trajectory_summary.csv")
OUTPUT_JSON = os.path.join(BASE_PATH, "grid_trajectory_summary.json")

# Optional: Save intermediate trajectories
SAVE_TRAJECTORIES = True
TRAJECTORIES_PATH = os.path.join(BASE_PATH, "trajectories.gpkg")

# SAMPLE MODE: Set to None for full data, or a number for testing
SAMPLE_ROWS = 1000000  # Test with 1 million rows first
# SAMPLE_ROWS = None  # Uncomment this for full processing

def create_trajectories(df):
    """
    Step 3: Create trajectories for each ship_id.
    Returns a GeoDataFrame with ship_id and LineString geometry.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Creating Trajectories")
    print("=" * 70)
    
    # Sort by ship_id and time
    print("Sorting data by ship_id and time...")
    df_sorted = df.sort_values(['decoded_ship_id', 'recv_dt']).copy()
    
    trajectories = []
    
    # Group by ship_id
    grouped = df_sorted.groupby('decoded_ship_id')
    total_ships = len(grouped)
    
    print(f"Processing {total_ships:,} unique ships...")
    
    skipped_single_point = 0
    skipped_errors = 0
    
    for idx, (ship_id, group) in enumerate(grouped, 1):
        if idx % 1000 == 0:
            print(f"  Progress: {idx:,}/{total_ships:,} ships ({idx/total_ships*100:.1f}%)")
        
        # Need at least 2 points to create a line
        if len(group) < 2:
            skipped_single_point += 1
            continue
        
        # Create points
        try:
            points = [Point(row['lon_val'], row['lat_val']) for _, row in group.iterrows()]
            
            # Create LineString
            line = LineString(points)
            
            trajectories.append({
                'ship_id': ship_id,
                'geometry': line,
                'point_count': len(points),
                'start_time': group['recv_dt'].min(),
                'end_time': group['recv_dt'].max()
            })
        except Exception as e:
            skipped_errors += 1
            if skipped_errors <= 10:  # Only print first 10 errors
                print(f"  Warning: Error creating trajectory for ship {ship_id}: {e}")
            continue
    
    print(f"\n✓ Successfully created {len(trajectories):,} trajectories")
    print(f"  - Ships with single point (skipped): {skipped_single_point:,}")
    print(f"  - Errors during creation: {skipped_errors:,}")
    
    # Convert to GeoDataFrame
    gdf_trajectories = gpd.GeoDataFrame(trajectories, crs='EPSG:4326')
    
    return gdf_trajectories

def load_grid(grid_path):
    """
    Load grid polygons from CSV with WKT format.
    """
    print("\n" + "=" * 70)
    print("Loading Grid Data")
    print("=" * 70)
    
    grid_df = pd.read_csv(grid_path)
    print(f"Loaded {len(grid_df):,} grid cells from CSV")
    
    # Parse WKT geometry
    print("Parsing WKT geometries...")
    grid_df['geometry'] = grid_df['wkt'].apply(wkt.loads)
    
    # Convert to GeoDataFrame
    gdf_grid = gpd.GeoDataFrame(grid_df, geometry='geometry', crs='EPSG:4326')
    
    # Create spatial index for faster intersection
    print("Creating spatial index for grid...")
    gdf_grid.sindex  # This creates the spatial index
    
    print(f"✓ Grid ready with {len(gdf_grid):,} cells")
    
    return gdf_grid

def intersect_trajectories_with_grid(gdf_trajectories, gdf_grid):
    """
    Step 4: Intersect trajectories with grid cells.
    Returns a GeoDataFrame with grid_id, ship_id, and intersected geometry.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Intersecting Trajectories with Grid")
    print("=" * 70)
    
    results = []
    total_trajectories = len(gdf_trajectories)
    
    print(f"Processing {total_trajectories:,} trajectories...")
    
    # Use spatial index for faster lookup
    spatial_index = gdf_grid.sindex
    
    for idx, traj_row in gdf_trajectories.iterrows():
        if (idx + 1) % 500 == 0:
            print(f"  Progress: {idx + 1:,}/{total_trajectories:,} ({(idx+1)/total_trajectories*100:.1f}%)")
        
        ship_id = traj_row['ship_id']
        trajectory = traj_row['geometry']
        
        # Get candidate grids using spatial index (bounding box intersection)
        possible_matches_idx = list(spatial_index.intersection(trajectory.bounds))
        possible_matches = gdf_grid.iloc[possible_matches_idx]
        
        # Precise intersection check
        intersecting_grids = possible_matches[possible_matches.intersects(trajectory)]
        
        for _, grid_row in intersecting_grids.iterrows():
            grid_id = grid_row['MIN1']
            grid_geom = grid_row['geometry']
            
            # Calculate intersection
            try:
                intersection = trajectory.intersection(grid_geom)
                
                # Only keep if intersection is a LineString or MultiLineString
                if intersection.is_empty:
                    continue
                
                if intersection.geom_type == 'LineString':
                    results.append({
                        'grid_id': grid_id,
                        'ship_id': ship_id,
                        'geometry': intersection
                    })
                elif intersection.geom_type == 'MultiLineString':
                    # Split MultiLineString into individual LineStrings
                    for line in intersection.geoms:
                        results.append({
                            'grid_id': grid_id,
                            'ship_id': ship_id,
                            'geometry': line
                        })
                elif intersection.geom_type == 'GeometryCollection':
                    # Extract only LineStrings from collection
                    for geom in intersection.geoms:
                        if geom.geom_type == 'LineString':
                            results.append({
                                'grid_id': grid_id,
                                'ship_id': ship_id,
                                'geometry': geom
                            })
            except Exception as e:
                print(f"  Warning: Error intersecting trajectory {ship_id} with grid {grid_id}: {e}")
                continue
    
    print(f"\n✓ Found {len(results):,} trajectory segments across grids")
    
    if not results:
        print("⚠️  No intersections found!")
        return gpd.GeoDataFrame()
    
    gdf_intersections = gpd.GeoDataFrame(results, crs='EPSG:4326')
    
    return gdf_intersections

def calculate_summary(gdf_intersections):
    """
    Step 5: Calculate summary statistics per grid.
    - 선박선길이sum: Total length of all trajectory segments in the grid (meters)
    - 선의갯수: Number of trajectory segments in the grid
    """
    print("\n" + "=" * 70)
    print("STEP 5: Calculating Summary Statistics")
    print("=" * 70)
    
    # Convert to projected CRS for accurate length calculation (Web Mercator)
    print("Converting to projected CRS for length calculation...")
    gdf_intersections_proj = gdf_intersections.to_crs('EPSG:3857')  # Web Mercator in meters
    
    # Calculate length for each segment
    print("Calculating segment lengths...")
    gdf_intersections_proj['length_m'] = gdf_intersections_proj.geometry.length
    
    # Group by grid_id
    print("Aggregating by grid cell...")
    summary = gdf_intersections_proj.groupby('grid_id').agg(
        선박선길이sum=('length_m', 'sum'),
        선의갯수=('length_m', 'count')
    ).reset_index()
    
    # Rename grid_id to 격자명
    summary.rename(columns={'grid_id': '격자명'}, inplace=True)
    
    # Round values
    summary['선박선길이sum'] = summary['선박선길이sum'].round(2)
    
    # Sort by grid name
    summary = summary.sort_values('격자명').reset_index(drop=True)
    
    print(f"\n✓ Summary calculated for {len(summary):,} grid cells")
    print(f"  - Total trajectory length: {summary['선박선길이sum'].sum():,.2f} meters")
    print(f"  - Total trajectory segments: {summary['선의갯수'].sum():,}")
    
    return summary

def save_results(summary_df, csv_path, json_path):
    """
    Save results to CSV and JSON formats.
    """
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    
    # Save to CSV
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ CSV saved: {csv_path}")
    
    # Save to JSON
    summary_dict = summary_df.to_dict(orient='records')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON saved: {json_path}")
    
    # Display sample
    print("\nSample results (first 10 rows):")
    print(summary_df.head(10).to_string(index=False))

def main():
    """
    Main processing pipeline for Phase 2.
    """
    start_time = datetime.now()
    
    print("=" * 70)
    print("PHASE 2: Trajectory Analysis and Grid Intersection")
    print("=" * 70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"\n❌ ERROR: Input file not found: {INPUT_PATH}")
        print("   Please run phase1_data_filtering.py first!")
        return
    
    # Load filtered data
    print(f"\nLoading filtered AIS data from Phase 1...")
    try:
        df_filtered = pd.read_csv(INPUT_PATH, low_memory=False)
        print(f"✓ Loaded {len(df_filtered):,} rows")
    except Exception as e:
        print(f"❌ ERROR loading input file: {e}")
        return
    
    if df_filtered.empty:
        print("❌ ERROR: Input file is empty!")
        return
    
    # Step 3: Create trajectories
    gdf_trajectories = create_trajectories(df_filtered)
    
    if gdf_trajectories.empty:
        print("\n❌ ERROR: No trajectories created. Exiting.")
        return
    
    # Optionally save trajectories
    if SAVE_TRAJECTORIES:
        print(f"\nSaving trajectories to {TRAJECTORIES_PATH}...")
        gdf_trajectories.to_file(TRAJECTORIES_PATH, driver='GPKG')
        print("✓ Trajectories saved")
    
    # Load grid
    try:
        gdf_grid = load_grid(GRID_PATH)
    except Exception as e:
        print(f"\n❌ ERROR loading grid file: {e}")
        return
    
    # Step 4: Intersect with grid
    gdf_intersections = intersect_trajectories_with_grid(gdf_trajectories, gdf_grid)
    
    if gdf_intersections.empty:
        print("\n❌ ERROR: No intersections found. Check if trajectories and grid overlap.")
        return
    
    # Step 5: Calculate summary
    summary_df = calculate_summary(gdf_intersections)
    
    # Save results
    save_results(summary_df, OUTPUT_CSV, OUTPUT_JSON)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE!")
    print("=" * 70)
    print(f"Start time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:    {duration}")
    print(f"\nOutput files:")
    print(f"  - CSV:  {OUTPUT_CSV}")
    print(f"  - JSON: {OUTPUT_JSON}")
    if SAVE_TRAJECTORIES:
        print(f"  - Trajectories: {TRAJECTORIES_PATH}")

if __name__ == "__main__":
    main()