"""
PHASE 2 (KD-Tree + Clip): 궤적-격자 매칭 초초고속 처리
- 방법 1: KD-Tree로 근처 격자만 찾기 (100배 빠름)
- 방법 2: Clip으로 자르기 (intersection보다 빠름)
- 결합: KD-Tree + Clip = 최고 속도!

처리:
1. AIS 점 → 선 (궤적) 변환
2. KD-Tree로 근처 격자 찾기
3. Clip으로 선 자르기
4. 격자별: 선분 개수 + 길이 합
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely import wkt, clip_by_rect
import json
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
import glob
import numpy as np
from scipy.spatial import cKDTree
import gc
import psutil

# --- File Paths ---
BASE_PATH = r"/media/data1/cmyoon/AIS_process"
INPUT_DIR = os.path.join(BASE_PATH, "filtered_daily")
GRID_PATH = os.path.join(BASE_PATH, "grid_polygon_wkt_new.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "results")
DAILY_RESULTS_DIR = os.path.join(OUTPUT_DIR, "daily_grids")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DAILY_RESULTS_DIR, exist_ok=True)

# 동적 프로세스 수
def get_optimal_processes():
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    if available_gb > 100:
        return 10
    elif available_gb > 50:
        return 5
    elif available_gb > 30:
        return 3
    else:
        return 1

NUM_PROCESSES = get_optimal_processes()

def create_trajectories(df, grid_bounds=None):
    """
    AIS 점 → 선 (궤적) 변환
    grid_bounds: (minx, miny, maxx, maxy) 격자 전체 범위 - 이 범위를 벗어나는 궤적은 필터링
    """
    df_sorted = df.sort_values(['decoded_ship_id', 'recv_dt']).copy()
    trajectories = []
    filtered_count = 0

    for ship_id, group in df_sorted.groupby('decoded_ship_id'):
        if len(group) < 2:
            continue
        try:
            points = [Point(row['lon_val'], row['lat_val']) for _, row in group.iterrows()]
            line = LineString(points)

            if line.length > 0:
                # 격자 범위 필터링
                if grid_bounds is not None:
                    minx, miny, maxx, maxy = grid_bounds
                    line_bounds = line.bounds  # (minx, miny, maxx, maxy)

                    # 궤적이 격자 범위를 벗어나는지 확인
                    if (line_bounds[0] < minx or line_bounds[1] < miny or
                        line_bounds[2] > maxx or line_bounds[3] > maxy):
                        filtered_count += 1
                        continue

                trajectories.append({
                    'ship_id': ship_id,
                    'geometry': line
                })
        except:
            continue

    if filtered_count > 0:
        print(f"  Filtered {filtered_count:,} trajectories outside grid bounds")

    return gpd.GeoDataFrame(trajectories, crs='EPSG:4326')

def build_grid_kdtree(gdf_grid):
    """KD-Tree 구축"""
    centroids = np.array([[geom.centroid.x, geom.centroid.y] 
                          for geom in gdf_grid.geometry])
    kdtree = cKDTree(centroids)
    return kdtree, centroids

def process_trajectory_with_kdtree(trajectory, ship_id, kdtree, gdf_grid, grid_bounds, search_radius=0.1):
    """
    ⭐ 최적화: KD-Tree + Clip 조합
    
    1. KD-Tree로 근처 격자만 찾기 (1,000개 중 10개)
    2. Clip으로 선 자르기
    """
    results = []
    
    # 궤적의 중심점으로 KD-Tree 검색
    traj_centroid = trajectory.centroid
    search_point = np.array([[traj_centroid.x, traj_centroid.y]])
    
    # ⭐ KD-Tree 검색: 근처 격자 찾기 (매우 빠름!)
    candidate_indices = kdtree.query_ball_point(search_point[0], search_radius)
    
    if not candidate_indices:
        return results
    
    # 각 근처 격자에 대해 Clip 적용
    for grid_idx in candidate_indices:
        try:
            grid_geom = gdf_grid.iloc[grid_idx].geometry
            grid_id = gdf_grid.iloc[grid_idx]['MIN1']
            grid_bounds_val = grid_bounds[grid_idx]
            
            # ⭐ Clip으로 선 자르기 (intersection보다 빠름!)
            minx, miny, maxx, maxy = grid_bounds_val
            clipped = clip_by_rect(trajectory, minx, miny, maxx, maxy)
            
            if clipped.is_empty:
                continue
            
            # 결과 처리
            if clipped.geom_type == 'LineString':
                results.append({
                    'grid_id': grid_id,
                    'ship_id': ship_id,
                    'geometry': clipped,
                    'length': clipped.length
                })
            elif clipped.geom_type == 'MultiLineString':
                for line in clipped.geoms:
                    if line.length > 0:
                        results.append({
                            'grid_id': grid_id,
                            'ship_id': ship_id,
                            'geometry': line,
                            'length': line.length
                        })
            elif clipped.geom_type == 'GeometryCollection':
                for geom in clipped.geoms:
                    if geom.geom_type == 'LineString' and geom.length > 0:
                        results.append({
                            'grid_id': grid_id,
                            'ship_id': ship_id,
                            'geometry': geom,
                            'length': geom.length
                        })
        except:
            continue
    
    return results

def process_single_day(args):
    """단일 날짜 처리"""
    day_file, gdf_grid, kdtree, grid_bounds, all_grid_ids, total_bounds = args
    day_num = os.path.basename(day_file).replace('filtered_', '').replace('.csv', '')
    
    output_file = os.path.join(DAILY_RESULTS_DIR, f"grid_summary_{day_num}.csv")
    if os.path.exists(output_file):
        print(f"[Day {day_num}] ✓ Already processed")
        return output_file
    
    print(f"[Day {day_num}] Starting...")
    start_time = datetime.now()
    
    try:
        # 1. AIS 데이터 로드
        df = pd.read_csv(day_file, low_memory=False)
        if df.empty:
            empty_df = pd.DataFrame({
                '격자명': all_grid_ids,
                '선의개수': 0,
                '선의길이합': 0.0
            })
            empty_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[Day {day_num}] No data")
            return output_file
        
        print(f"[Day {day_num}] Loaded {len(df):,} AIS points")

        # 2. 점 → 선 변환 (격자 범위 필터링 적용)
        gdf_traj = create_trajectories(df, grid_bounds=total_bounds)
        if gdf_traj.empty:
            empty_df = pd.DataFrame({
                '격자명': all_grid_ids,
                '선의개수': 0,
                '선의길이합': 0.0
            })
            empty_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[Day {day_num}] No trajectories")
            return output_file
        
        print(f"[Day {day_num}] Created {len(gdf_traj):,} trajectories")
        
        # 3. KD-Tree + Clip 처리
        all_results = []
        
        for idx, (traj_idx, traj_row) in enumerate(gdf_traj.iterrows()):
            if (idx + 1) % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(gdf_traj) - idx - 1) / rate if rate > 0 else 0
                
                print(f"  [{idx + 1}/{len(gdf_traj)}] ({(idx+1)/len(gdf_traj)*100:.1f}%) "
                      f"| {rate:.1f} traj/sec | ETA: {remaining/60:.1f} min | "
                      f"Segments: {len(all_results):,}")
            
            ship_id = traj_row['ship_id']
            trajectory = traj_row['geometry']
            
            # ⭐ KD-Tree + Clip으로 처리
            results = process_trajectory_with_kdtree(
                trajectory, ship_id, kdtree, gdf_grid, grid_bounds, 
                search_radius=0.15  # 반경 조정 가능
            )
            
            if results:
                all_results.extend(results)
            
            # 주기적 메모리 정리
            if (idx + 1) % 500 == 0:
                gc.collect()
        
        if not all_results:
            empty_df = pd.DataFrame({
                '격자명': all_grid_ids,
                '선의개수': 0,
                '선의길이합': 0.0
            })
            empty_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[Day {day_num}] No clipped segments")
            return output_file
        
        print(f"[Day {day_num}] Found {len(all_results):,} clipped segments")
        
        # 4. 길이 계산 (WGS84 → Web Mercator)
        gdf_int = gpd.GeoDataFrame(all_results, crs='EPSG:4326')
        gdf_int_proj = gdf_int.to_crs('EPSG:3857')
        gdf_int_proj['length_m'] = gdf_int_proj.geometry.length
        
        # 5. 격자별 집계
        summary = gdf_int_proj.groupby('grid_id').agg(
            length_sum=('length_m', 'sum'),
            count=('grid_id', 'count')
        ).reset_index()
        
        # 6. 전체 격자와 병합
        all_grids_df = pd.DataFrame({'grid_id': all_grid_ids})
        full_summary = all_grids_df.merge(summary, on='grid_id', how='left')
        full_summary['length_sum'] = full_summary['length_sum'].fillna(0.0)
        full_summary['count'] = full_summary['count'].fillna(0).astype(int)
        
        full_summary.rename(columns={
            'grid_id': '격자명',
            'count': '선의개수',
            'length_sum': '선의길이합'
        }, inplace=True)
        
        full_summary['선의길이합'] = full_summary['선의길이합'].round(2)
        full_summary.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        data_grids = (full_summary['선의개수'] > 0).sum()
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"[Day {day_num}] ✓ Complete!")
        print(f"  - Time: {elapsed:.1f} sec ({elapsed/60:.1f} min)")
        print(f"  - Grids with data: {data_grids:,}")
        print(f"  - Total length: {full_summary['선의길이합'].sum():,.0f} m")
        
        return output_file
        
    except Exception as e:
        print(f"[Day {day_num}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 파이프라인"""
    start_time = datetime.now()
    
    print("=" * 70)
    print("PHASE 2: KD-Tree + Clip (최고 성능)")
    print("=" * 70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.available/(1024**3):.1f}GB available ({mem.percent}%)")
    print(f"Using: {NUM_PROCESSES} processes")
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ERROR: {INPUT_DIR} not found")
        return
    
    daily_files = sorted(glob.glob(os.path.join(INPUT_DIR, "filtered_*.csv")))
    if not daily_files:
        print(f"❌ ERROR: No files in {INPUT_DIR}")
        return
    
    print(f"✓ Found {len(daily_files)} daily files")
    
    # 격자 로드
    print(f"\nLoading grid...")
    try:
        grid_df = pd.read_csv(GRID_PATH)
        grid_df['wkt'] = grid_df['wkt'].str.replace(';', ',')
        grid_df['geometry'] = grid_df['wkt'].apply(wkt.loads)
        gdf_grid = gpd.GeoDataFrame(grid_df, geometry='geometry', crs='EPSG:4326')
        
        all_grid_ids = gdf_grid['MIN1'].tolist()
        print(f"✓ Loaded {len(gdf_grid):,} grid cells")
        
        # KD-Tree 구축
        print(f"Building KD-Tree...")
        kdtree, _ = build_grid_kdtree(gdf_grid)
        print(f"✓ KD-Tree ready")
        
        # Bbox 사전 계산
        print(f"Precomputing bounds...")
        grid_bounds = {i: geom.bounds for i, geom in enumerate(gdf_grid.geometry)}
        print(f"✓ Bounds ready")

        # 전체 격자 범위 계산 (궤적 필터링용)
        total_bounds = gdf_grid.total_bounds  # (minx, miny, maxx, maxy)
        print(f"Total grid bounds: ({total_bounds[0]:.2f}, {total_bounds[1]:.2f}, {total_bounds[2]:.2f}, {total_bounds[3]:.2f})")
        
    except Exception as e:
        print(f"❌ ERROR loading grid: {e}")
        return
    
    # 처리
    print(f"\n{'=' * 70}")
    print(f"Processing {len(daily_files)} days...")
    print(f"{'=' * 70}\n")

    args_list = [(f, gdf_grid, kdtree, grid_bounds, all_grid_ids, total_bounds) for f in daily_files]
    
    with Pool(NUM_PROCESSES) as pool:
        result_files = pool.map(process_single_day, args_list)
    
    valid_files = [f for f in result_files if f is not None]
    
    # 병합
    print(f"\n{'=' * 70}")
    print(f"Merging {len(valid_files)} daily files...")
    print(f"{'=' * 70}")
    
    daily_summaries = []
    for result_file in valid_files:
        try:
            df = pd.read_csv(result_file)
            daily_summaries.append(df)
            data_count = (df['선의개수'] > 0).sum()
            print(f"  {os.path.basename(result_file)}: {data_count:,}/{len(df):,} grids with data")
        except Exception as e:
            print(f"  Warning: {result_file}: {e}")
    
    if not daily_summaries:
        print("❌ No summaries")
        return
    
    # 최종 집계
    combined_df = pd.concat(daily_summaries, ignore_index=True)
    final_summary = combined_df.groupby('격자명').agg(
        선의길이합=('선의길이합', 'sum'),
        선의개수=('선의개수', 'sum')
    ).reset_index()
    
    final_summary['선의길이합'] = final_summary['선의길이합'].round(2)
    final_summary['선의개수'] = final_summary['선의개수'].astype(int)
    final_summary = final_summary.sort_values('격자명').reset_index(drop=True)
    
    # 저장
    FINAL_CSV = os.path.join(OUTPUT_DIR, "grid_trajectory_summary.csv")
    FINAL_JSON = os.path.join(OUTPUT_DIR, "grid_trajectory_summary.json")
    
    final_summary.to_csv(FINAL_CSV, index=False, encoding='utf-8-sig')
    with open(FINAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_summary.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Final CSV: {FINAL_CSV}")
    print(f"✓ Final JSON: {FINAL_JSON}")
    
    print(f"\nResults:")
    print(f"  Total grids: {len(final_summary):,}")
    print(f"  Grids with data: {(final_summary['선의개수'] > 0).sum():,}")
    print(f"  Total segments: {final_summary['선의개수'].sum():,}")
    print(f"  Total length: {final_summary['선의길이합'].sum():,.0f} m")
    
    end_time = datetime.now()
    print(f"\nDuration: {end_time - start_time}")

if __name__ == "__main__":
    main()
