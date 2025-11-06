"""
PHASE 2 (KD-Tree + 날짜별 저장 + SHP 출력): 궤적-격자 매칭 초고속 처리
- 날짜별 즉시 저장 (중단 후 재개 가능)
- 전체 격자 포함 (데이터 없으면 0)
- ⭐ 최적화: GeoDataFrame 인덱싱 오버헤드 제거
- ⭐ SHP 출력: 격자로 잘린 궤적을 GIS 포맷으로 저장
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
GRID_PATH = os.path.join(BASE_PATH, "grid_polygon_wkt.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "results")
DAILY_RESULTS_DIR = os.path.join(OUTPUT_DIR, "daily_grids")
DAILY_SHP_DIR = os.path.join(OUTPUT_DIR, "daily_trajectories_shp")  # ⭐ 새로 추가
FINAL_CSV = os.path.join(OUTPUT_DIR, "grid_trajectory_summary.csv")
FINAL_JSON = os.path.join(OUTPUT_DIR, "grid_trajectory_summary.json")
FINAL_SHP_DIR = os.path.join(OUTPUT_DIR, "all_trajectories_shp")  # ⭐ 새로 추가

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DAILY_RESULTS_DIR, exist_ok=True)
os.makedirs(DAILY_SHP_DIR, exist_ok=True)  # ⭐ 새로 추가
os.makedirs(FINAL_SHP_DIR, exist_ok=True)  # ⭐ 새로 추가

# 메모리 효율: 10개 프로세스 (503GB 메모리 고려)
NUM_PROCESSES = 10  # 안전한 개수

# ⭐ SHP 출력 옵션
SAVE_DAILY_SHP = True  # 날짜별 SHP 저장
SAVE_FINAL_SHP = True  # 전체 통합 SHP 저장

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
    """격자의 중심점으로 KD-Tree 생성"""
    centroids = np.array([[geom.centroid.x, geom.centroid.y] 
                          for geom in gdf_grid.geometry])
    kdtree = cKDTree(centroids)
    return kdtree, centroids

def sample_trajectory_points(trajectory, max_distance=0.05):
    """궤적을 균등하게 샘플링"""
    if trajectory.length == 0:
        return []
    
    num_samples = max(10, int(trajectory.length / max_distance))
    num_samples = min(num_samples, 1000)
    
    distances = np.linspace(0, trajectory.length, num_samples)
    points = [trajectory.interpolate(d) for d in distances]
    
    return np.array([[p.x, p.y] for p in points])

def intersect_with_grid_kdtree_optimized(gdf_trajectories, gdf_grid, kdtree, grid_lookup, grid_bounds):
    """
    ⭐ 최적화 버전: KD-Tree를 이용한 초고속 궤적-격자 매칭
    
    주요 개선:
    1. grid_lookup 딕셔너리로 O(1) 접근
    2. grid_bounds 미리 계산
    3. 배치 처리로 메모리 관리
    """
    if gdf_trajectories.empty or gdf_grid.empty:
        return gpd.GeoDataFrame()
    
    print(f"    Processing trajectories with KD-Tree (OPTIMIZED)...")
    start_time = datetime.now()
    
    results = []
    total_trajectories = len(gdf_trajectories)
    
    for idx, traj_row in gdf_trajectories.iterrows():
        # 진행 상황 출력 (100개마다)
        if (idx + 1) % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_trajectories - idx - 1) / rate if rate > 0 else 0
            
            print(f"    [{idx + 1}/{total_trajectories}] ({(idx+1)/total_trajectories*100:.1f}%) "
                  f"| {rate:.1f} traj/sec | ETA: {remaining/60:.1f} min | "
                  f"Segments: {len(results):,}")
        
        ship_id = traj_row['ship_id']
        trajectory = traj_row['geometry']
        
        # 궤적 샘플링
        sampled_points = sample_trajectory_points(trajectory)
        if len(sampled_points) == 0:
            continue
        
        # KD-Tree로 후보 격자 찾기
        search_radius = 0.1
        candidate_grid_indices = set()
        for point in sampled_points:
            nearby_indices = kdtree.query_ball_point(point, search_radius)
            candidate_grid_indices.update(nearby_indices)
        
        if not candidate_grid_indices:
            continue
        
        # 바운딩 박스 필터
        traj_bounds = trajectory.bounds
        filtered_candidates = []
        for grid_idx in candidate_grid_indices:
            gb = grid_bounds[grid_idx]
            if not (gb[2] < traj_bounds[0] or gb[0] > traj_bounds[2] or
                    gb[3] < traj_bounds[1] or gb[1] > traj_bounds[3]):
                filtered_candidates.append(grid_idx)
        
        if not filtered_candidates:
            continue
        
        # ⭐ 개선: grid_lookup으로 O(1) 접근
        for grid_idx in filtered_candidates:
            grid_info = grid_lookup[grid_idx]
            grid_geom = grid_info['geometry']
            grid_id = grid_info['grid_id']
            
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
    """Process a single day and save with ALL grids."""
    day_file, gdf_grid, kdtree, all_grid_ids, grid_lookup, grid_bounds = args
    day_num = os.path.basename(day_file).replace('filtered_', '').replace('.csv', '')
    
    # 이미 처리된 날짜는 스킵
    output_file = os.path.join(DAILY_RESULTS_DIR, f"grid_summary_{day_num}.csv")
    if os.path.exists(output_file):
        print(f"[Day {day_num}] ✓ Already processed, skipping")
        return output_file
    
    print(f"[Day {day_num}] Starting...")
    
    try:
        df = pd.read_csv(day_file, low_memory=False)
        if df.empty:
            print(f"[Day {day_num}] No data")
            empty_summary = pd.DataFrame({
                '격자명': all_grid_ids,
                '선박선길이sum': 0.0,
                '선의갯수': 0
            })
            empty_summary.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[Day {day_num}] ✓ Saved empty grid (all zeros)")
            return output_file
        
        print(f"[Day {day_num}] Loaded {len(df):,} rows")
        
        gdf_traj = create_trajectories(df)
        if gdf_traj.empty:
            print(f"[Day {day_num}] No trajectories")
            empty_summary = pd.DataFrame({
                '격자명': all_grid_ids,
                '선박선길이sum': 0.0,
                '선의갯수': 0
            })
            empty_summary.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[Day {day_num}] ✓ Saved empty grid")
            return output_file
        
        print(f"[Day {day_num}] Created {len(gdf_traj):,} trajectories")
        
        # ⭐ 최적화된 KD-Tree 기반 교차 계산
        gdf_int = intersect_with_grid_kdtree_optimized(gdf_traj, gdf_grid, kdtree, grid_lookup, grid_bounds)
        
        if gdf_int.empty:
            print(f"[Day {day_num}] No intersections")
            empty_summary = pd.DataFrame({
                '격자명': all_grid_ids,
                '선박선길이sum': 0.0,
                '선의갯수': 0
            })
            empty_summary.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[Day {day_num}] ✓ Saved empty grid")
            return output_file
        
        print(f"[Day {day_num}] Found {len(gdf_int):,} segments")
        
        # ⭐ SHP 파일 저장 (날짜별)
        if SAVE_DAILY_SHP:
            shp_output = os.path.join(DAILY_SHP_DIR, f"trajectories_{day_num}.shp")
            try:
                # EPSG:3857로 투영하여 저장 (미터 단위로 거리 계산하기 좋음)
                gdf_int_proj = gdf_int.to_crs('EPSG:3857')
                gdf_int_proj['length_m'] = gdf_int_proj.geometry.length
                
                # 속성 추가
                gdf_int_proj['length_m'] = gdf_int_proj['length_m'].round(2)
                
                # SHP 저장 (한글 컬럼명은 자동으로 ASCII로 변환됨)
                gdf_int_proj.to_file(shp_output, driver='SHAPEFILE', encoding='utf-8')
                print(f"[Day {day_num}] ✓ Saved SHP: {shp_output} ({len(gdf_int_proj):,} features)")
            except Exception as e:
                print(f"[Day {day_num}] ⚠️ SHP save failed: {e}")
        
        # 길이 계산 (CSV용)
        gdf_int_proj = gdf_int.to_crs('EPSG:3857')
        gdf_int_proj['length_m'] = gdf_int_proj.geometry.length
        
        # 격자별 집계
        summary = gdf_int_proj.groupby('grid_id').agg(
            length_sum=('length_m', 'sum'),
            count=('length_m', 'count')
        ).reset_index()
        
        # ⭐ 전체 격자와 병합 (데이터 없는 격자는 0으로)
        all_grids_df = pd.DataFrame({'grid_id': all_grid_ids})
        full_summary = all_grids_df.merge(summary, on='grid_id', how='left')
        full_summary['length_sum'] = full_summary['length_sum'].fillna(0)
        full_summary['count'] = full_summary['count'].fillna(0).astype(int)
        
        # 컬럼명 변경
        full_summary.rename(columns={
            'grid_id': '격자명',
            'length_sum': '선박선길이sum',
            'count': '선의갯수'
        }, inplace=True)
        
        full_summary['선박선길이sum'] = full_summary['선박선길이sum'].round(2)
        
        # 저장
        full_summary.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        data_grids = (full_summary['선의갯수'] > 0).sum()
        print(f"[Day {day_num}] ✓ Complete: {data_grids:,}/{len(all_grid_ids):,} grids with data")
        print(f"[Day {day_num}] ✓ Saved to {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"[Day {day_num}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main pipeline."""
    start_time = datetime.now()
    
    print("=" * 70)
    print("PHASE 2: KD-Tree with Daily Save [OPTIMIZED] + SHP Export")
    print("=" * 70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU cores: {cpu_count()}, Using: {NUM_PROCESSES} processes")
    print(f"Daily SHP export: {'✓ Enabled' if SAVE_DAILY_SHP else '✗ Disabled'}")
    print(f"Final SHP export: {'✓ Enabled' if SAVE_FINAL_SHP else '✗ Disabled'}")
    
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
        
        # 전체 격자 ID 리스트 추출
        all_grid_ids = gdf_grid['MIN1'].tolist()
        
        print(f"✓ Loaded {len(gdf_grid):,} total grid cells")
        
        # Build KD-Tree
        print(f"Building KD-Tree...")
        kdtree, centroids = build_grid_kdtree(gdf_grid)
        print(f"✓ KD-Tree ready")
        
        # ⭐ 핵심 최적화: grid lookup 딕셔너리 생성 (O(1) 접근)
        print(f"Building grid lookup index (for O(1) access)...")
        grid_lookup = {
            i: {
                'geometry': gdf_grid.iloc[i].geometry,
                'grid_id': gdf_grid.iloc[i]['MIN1']
            }
            for i in range(len(gdf_grid))
        }
        print(f"✓ Grid lookup ready ({len(grid_lookup):,} grids indexed)")
        
        # ⭐ 바운딩 박스 미리 계산
        print(f"Precomputing grid bounds...")
        grid_bounds = {i: geom.bounds for i, geom in enumerate(gdf_grid.geometry)}
        print(f"✓ Grid bounds ready")
        
    except Exception as e:
        print(f"❌ ERROR loading grid: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process in parallel
    print(f"\n{'=' * 70}")
    print(f"Processing {len(daily_files)} days in parallel...")
    print(f"Each day saved with ALL {len(all_grid_ids):,} grids")
    print(f"{'=' * 70}\n")
    
    # ⭐ grid_lookup과 grid_bounds를 인자로 전달
    args_list = [(f, gdf_grid, kdtree, all_grid_ids, grid_lookup, grid_bounds) for f in daily_files]
    
    with Pool(NUM_PROCESSES) as pool:
        result_files = pool.map(process_single_day, args_list)
    
    valid_files = [f for f in result_files if f is not None]
    
    if not valid_files:
        print("❌ No valid results")
        return
    
    print(f"\n{'=' * 70}")
    print(f"Merging {len(valid_files)} daily files...")
    print(f"{'=' * 70}")
    
    # 날짜별 파일 병합
    daily_summaries = []
    all_shp_files = []
    
    for result_file in valid_files:
        try:
            df = pd.read_csv(result_file)
            daily_summaries.append(df)
            data_count = (df['선의갯수'] > 0).sum()
            print(f"  {os.path.basename(result_file)}: {data_count:,}/{len(df):,} grids with data")
            
            # ⭐ SHP 파일 수집
            if SAVE_DAILY_SHP:
                day_num = os.path.basename(result_file).replace('grid_summary_', '').replace('.csv', '')
                shp_file = os.path.join(DAILY_SHP_DIR, f"trajectories_{day_num}.shp")
                if os.path.exists(shp_file):
                    all_shp_files.append(shp_file)
        except Exception as e:
            print(f"  Warning: {result_file}: {e}")
    
    if not daily_summaries:
        print("❌ No summaries to merge")
        return
    
    combined_df = pd.concat(daily_summaries, ignore_index=True)
    
    # 전체 기간 집계
    final_summary = combined_df.groupby('격자명').agg(
        선박선길이sum=('선박선길이sum', 'sum'),
        선의갯수=('선의갯수', 'sum')
    ).reset_index()
    
    final_summary['선박선길이sum'] = final_summary['선박선길이sum'].round(2)
    final_summary['선의갯수'] = final_summary['선의갯수'].astype(int)
    final_summary = final_summary.sort_values('격자명').reset_index(drop=True)
    
    # Save final
    final_summary.to_csv(FINAL_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✓ Final CSV: {FINAL_CSV}")
    
    with open(FINAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_summary.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    print(f"✓ Final JSON: {FINAL_JSON}")
    
    # ⭐ 전체 SHP 병합 (선택사항)
    if SAVE_FINAL_SHP and all_shp_files:
        print(f"\nMerging all daily SHP files...")
        try:
            all_gdf = []
            for shp_file in sorted(all_shp_files):
                try:
                    gdf = gpd.read_file(shp_file)
                    all_gdf.append(gdf)
                    print(f"  Loaded: {os.path.basename(shp_file)} ({len(gdf):,} features)")
                except Exception as e:
                    print(f"  ⚠️ Failed to load {shp_file}: {e}")
            
            if all_gdf:
                # 전체 병합
                final_gdf = pd.concat(all_gdf, ignore_index=True)
                
                # 최종 SHP 저장
                final_shp_file = os.path.join(FINAL_SHP_DIR, "all_trajectories.shp")
                final_gdf.to_file(final_shp_file, driver='SHAPEFILE', encoding='utf-8')
                print(f"\n✓ Final merged SHP: {final_shp_file} ({len(final_gdf):,} features)")
        except Exception as e:
            print(f"⚠️ Failed to merge SHP files: {e}")
    
    print(f"\nSample (first 10 with data):")
    sample = final_summary[final_summary['선의갯수'] > 0].head(10)
    print(sample.to_string(index=False))
    
    end_time = datetime.now()
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Duration: {end_time - start_time}")
    print(f"Total grids: {len(final_summary):,}")
    print(f"Grids with data: {(final_summary['선의갯수'] > 0).sum():,}")
    print(f"Total length: {final_summary['선박선길이sum'].sum():,.2f} m")
    print(f"Total segments: {final_summary['선의갯수'].sum():,}")
    
    print(f"\n{'=' * 70}")
    print("📁 Output Folders:")
    print(f"{'=' * 70}")
    print(f"\n1️⃣  Daily CSV Results:")
    print(f"   {DAILY_RESULTS_DIR}")
    
    if SAVE_DAILY_SHP:
        print(f"\n2️⃣  Daily SHP Trajectories (날짜별 궤적):")
        print(f"   {DAILY_SHP_DIR}")
        print(f"   Files: {len(all_shp_files)} SHP files")
        print(f"   → QGis/ArcGis에서 직접 시각화 가능!")
    
    print(f"\n3️⃣  Summary Results:")
    print(f"   CSV: {FINAL_CSV}")
    print(f"   JSON: {FINAL_JSON}")
    
    if SAVE_FINAL_SHP:
        print(f"\n4️⃣  Final Merged SHP (전체 궤적):")
        print(f"   {FINAL_SHP_DIR}")
        print(f"   File: all_trajectories.shp")
        print(f"   → 전체 선박 궤적 한 번에 시각화!")
    
    print(f"\n{'=' * 70}")
    print("🔍 View in GIS:")
    print(f"{'=' * 70}")
    print(f"  QGIS에서 다음 파일들을 열 수 있습니다:")
    if SAVE_DAILY_SHP:
        print(f"  - {DAILY_SHP_DIR}/*.shp (날짜별)")
    if SAVE_FINAL_SHP:
        print(f"  - {FINAL_SHP_DIR}/all_trajectories.shp (전체)")

if __name__ == "__main__":
    main()