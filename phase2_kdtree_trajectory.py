"""
PHASE 2 (KD-Tree 최적화): 궤적-격자 매칭 초고속 처리
전략:
1. 궤적을 점들로 샘플링
2. KD-Tree로 가까운 격자 빠르게 찾기
3. 후보 격자에 대해서만 정확한 intersection 계산
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
import numpy as np
from scipy.spatial import cKDTree

# --- File Paths ---
BASE_PATH = r"/media/data1/cmyoon/AIS_process"
INPUT_DIR = os.path.join(BASE_PATH, "filtered_daily")
GRID_PATH = os.path.join(BASE_PATH, "grid_polygon_wkt.csv")  # 또는 _korea.csv
OUTPUT_DIR = os.path.join(BASE_PATH, "results")
FINAL_CSV = os.path.join(OUTPUT_DIR, "grid_trajectory_summary.csv")
FINAL_JSON = os.path.join(OUTPUT_DIR, "grid_trajectory_summary.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_PROCESSES = max(1, cpu_count() - 1)

def create_trajectories(df):
    """Create trajectories for each ship_id."""
    df_sorted = df.sort_values(['decoded_ship_id', 'recv_dt']).copy()
    trajectories = []
    
    for ship_id, group in df_sorted.groupby('decoded_ship_id'):
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
        except:
            continue
    
    return gpd.GeoDataFrame(trajectories, crs='EPSG:4326')

def build_grid_kdtree(gdf_grid):
    """
    격자의 중심점으로 KD-Tree 생성
    빠른 공간 검색을 위해
    """
    centroids = np.array([[geom.centroid.x, geom.centroid.y] 
                          for geom in gdf_grid.geometry])
    kdtree = cKDTree(centroids)
    return kdtree, centroids

def sample_trajectory_points(trajectory, max_distance=0.05):
    """
    궤적을 균등하게 샘플링
    max_distance: 샘플링 간격 (도 단위, 약 5km)
    """
    if trajectory.length == 0:
        return []
    
    # 궤적 길이에 따라 샘플 수 결정
    num_samples = max(10, int(trajectory.length / max_distance))
    num_samples = min(num_samples, 1000)  # 최대 1000개
    
    distances = np.linspace(0, trajectory.length, num_samples)
    points = [trajectory.interpolate(d) for d in distances]
    
    return np.array([[p.x, p.y] for p in points])

def intersect_with_grid_kdtree(gdf_trajectories, gdf_grid, kdtree):
    """
    KD-Tree를 이용한 초고속 궤적-격자 매칭
    
    전략:
    1. 궤적을 점들로 샘플링
    2. KD-Tree로 각 점 근처 격자 찾기 (O(log n))
    3. 후보 격자에 대해서만 정확한 intersection 계산
    """
    if gdf_trajectories.empty or gdf_grid.empty:
        return gpd.GeoDataFrame()
    
    print(f"    Building grid KD-Tree...")
    start_time = datetime.now()
    
    results = []
    total_trajectories = len(gdf_trajectories)
    
    # 격자별 바운딩 박스 미리 계산 (속도 향상)
    grid_bounds = {i: geom.bounds for i, geom in enumerate(gdf_grid.geometry)}
    
    print(f"    Processing trajectories with KD-Tree acceleration...")
    
    for idx, traj_row in gdf_trajectories.iterrows():
        if (idx + 1) % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_trajectories - idx - 1) / rate if rate > 0 else 0
            
            print(f"    [{idx + 1}/{total_trajectories}] ({(idx+1)/total_trajectories*100:.1f}%) "
                  f"| {rate:.1f} traj/sec | ETA: {remaining/60:.1f} min | "
                  f"Found: {len(results):,} segments")
        
        ship_id = traj_row['ship_id']
        trajectory = traj_row['geometry']
        
        # 1️⃣ 궤적을 점들로 샘플링
        sampled_points = sample_trajectory_points(trajectory)
        
        if len(sampled_points) == 0:
            continue
        
        # 2️⃣ KD-Tree로 각 점 근처 격자 빠르게 찾기
        # query_ball_point: 반경 내 모든 점 찾기
        search_radius = 0.1  # 약 10km (도 단위)
        
        candidate_grid_indices = set()
        for point in sampled_points:
            # 초고속 검색! O(log n)
            nearby_indices = kdtree.query_ball_point(point, search_radius)
            candidate_grid_indices.update(nearby_indices)
        
        if not candidate_grid_indices:
            continue
        
        # 3️⃣ 후보 격자에 대해서만 정확한 intersection 계산
        candidate_grids = gdf_grid.iloc[list(candidate_grid_indices)]
        
        # 추가 필터: 바운딩 박스 검사
        traj_bounds = trajectory.bounds
        filtered_candidates = []
        
        for grid_idx in candidate_grid_indices:
            gb = grid_bounds[grid_idx]
            # 바운딩 박스 오버랩 체크
            if not (gb[2] < traj_bounds[0] or gb[0] > traj_bounds[2] or
                    gb[3] < traj_bounds[1] or gb[1] > traj_bounds[3]):
                filtered_candidates.append(grid_idx)
        
        if not filtered_candidates:
            continue
        
        final_candidates = gdf_grid.iloc[filtered_candidates]
        
        # 4️⃣ 실제 교차 계산 (매우 적은 수의 격자만!)
        for grid_idx, grid_row in zip(filtered_candidates, final_candidates.itertuples()):
            grid_geom = gdf_grid.iloc[grid_idx].geometry
            grid_id = gdf_grid.iloc[grid_idx]['MIN1']
            
            try:
                if not trajectory.intersects(grid_geom):
                    continue
                
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
                        if line.length > 0:
                            results.append({
                                'grid_id': grid_id,
                                'ship_id': ship_id,
                                'geometry': line
                            })
                elif intersection.geom_type == 'GeometryCollection':
                    for geom in intersection.geoms:
                        if geom.geom_type == 'LineString' and geom.length > 0:
                            results.append({
                                'grid_id': grid_id,
                                'ship_id': ship_id,
                                'geometry': geom
                            })
            except:
                continue
    
    if not results:
        return gpd.GeoDataFrame()
    
    return gpd.GeoDataFrame(results, crs='EPSG:4326')

def process_single_day(args):
    """Process a single day's data."""
    day_file, gdf_grid, kdtree = args
    day_num = os.path.basename(day_file).replace('filtered_', '').replace('.csv', '')
    
    print(f"[Day {day_num}] Starting...")
    
    try:
        df = pd.read_csv(day_file, low_memory=False)
        if df.empty:
            print(f"[Day {day_num}] No data")
            return None
        
        print(f"[Day {day_num}] Loaded {len(df):,} rows")
        
        gdf_traj = create_trajectories(df)
        if gdf_traj.empty:
            print(f"[Day {day_num}] No trajectories")
            return None
        
        print(f"[Day {day_num}] Created {len(gdf_traj):,} trajectories")
        
        # KD-Tree 기반 교차 계산
        gdf_int = intersect_with_grid_kdtree(gdf_traj, gdf_grid, kdtree)
        if gdf_int.empty:
            print(f"[Day {day_num}] No intersections")
            return None
        
        print(f"[Day {day_num}] Found {len(gdf_int):,} segments")
        
        # 길이 계산
        gdf_int_proj = gdf_int.to_crs('EPSG:3857')
        gdf_int_proj['length_m'] = gdf_int_proj.geometry.length
        
        summary = gdf_int_proj.groupby('grid_id').agg(
            length_sum=('length_m', 'sum'),
            count=('length_m', 'count')
        ).reset_index()
        
        print(f"[Day {day_num}] ✓ Complete: {len(summary):,} grids")
        return summary
        
    except Exception as e:
        print(f"[Day {day_num}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main pipeline."""
    start_time = datetime.now()
    
    print("=" * 70)
    print("PHASE 2: KD-Tree Accelerated Trajectory Analysis")
    print("=" * 70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU cores: {cpu_count()}, Using: {NUM_PROCESSES} processes")
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ERROR: {INPUT_DIR} not found")
        return
    
    daily_files = sorted(glob.glob(os.path.join(INPUT_DIR, "filtered_*.csv")))
    if not daily_files:
        print(f"❌ ERROR: No files in {INPUT_DIR}")
        return
    
    print(f"✓ Found {len(daily_files)} daily files")
    
    # Load grid
    print(f"\nLoading grid...")
    try:
        grid_df = pd.read_csv(GRID_PATH)
        grid_df['wkt'] = grid_df['wkt'].str.replace(';', ',')
        grid_df['geometry'] = grid_df['wkt'].apply(wkt.loads)
        gdf_grid = gpd.GeoDataFrame(grid_df, geometry='geometry', crs='EPSG:4326')
        print(f"✓ Loaded {len(gdf_grid):,} grid cells")
        
        # Build KD-Tree
        print(f"Building KD-Tree for {len(gdf_grid):,} grid centroids...")
        kdtree, centroids = build_grid_kdtree(gdf_grid)
        print(f"✓ KD-Tree ready (O(log n) spatial search enabled)")
        
    except Exception as e:
        print(f"❌ ERROR loading grid: {e}")
        return
    
    # Process in parallel
    print(f"\n{'=' * 70}")
    print(f"Processing {len(daily_files)} days in parallel...")
    print(f"{'=' * 70}\n")
    
    args_list = [(f, gdf_grid, kdtree) for f in daily_files]
    
    with Pool(NUM_PROCESSES) as pool:
        results = pool.map(process_single_day, args_list)
    
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("❌ No valid results")
        return
    
    print(f"\nCombining {len(valid_results)} days...")
    combined_df = pd.concat(valid_results, ignore_index=True)
    
    final_summary = combined_df.groupby('grid_id').agg(
        선박선길이sum=('length_sum', 'sum'),
        선의갯수=('count', 'sum')
    ).reset_index()
    
    final_summary.rename(columns={'grid_id': '격자명'}, inplace=True)
    final_summary['선박선길이sum'] = final_summary['선박선길이sum'].round(2)
    final_summary['선의갯수'] = final_summary['선의갯수'].astype(int)
    final_summary = final_summary.sort_values('격자명').reset_index(drop=True)
    
    # Save
    final_summary.to_csv(FINAL_CSV, index=False, encoding='utf-8-sig')
    print(f"✓ CSV: {FINAL_CSV}")
    
    with open(FINAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_summary.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    print(f"✓ JSON: {FINAL_JSON}")
    
    print(f"\nSample (first 10):")
    print(final_summary.head(10).to_string(index=False))
    
    end_time = datetime.now()
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Duration: {end_time - start_time}")
    print(f"Grids: {len(final_summary):,}")
    print(f"Total length: {final_summary['선박선길이sum'].sum():,.2f} m")
    print(f"Total segments: {final_summary['선의갯수'].sum():,}")

if __name__ == "__main__":
    main()