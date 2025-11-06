# AIS Data Processing Pipeline

선박 자동식별장치(AIS) 데이터를 처리하여 궤적을 분석하고 격자 기반 통계를 생성하는 파이프라인입니다.

## 프로젝트 개요

이 프로젝트는 대용량 AIS 데이터를 효율적으로 처리하여 선박 궤적을 추출하고, 격자(Grid) 기반으로 통행량을 분석합니다. KD-Tree와 Clip 알고리즘을 활용한 고속 지리공간 처리를 구현했습니다.

## 주요 기능

- **Phase 1: 데이터 필터링**
  - Base64로 인코딩된 선박 ID 디코딩
  - 선박 정보(vessel_info)와 매칭
  - 선박 종류 필터링 (화물선 70-79, 탱커선 80-89)
  - 유효하지 않은 좌표 제거
  - 청크 단위 병렬 처리로 메모리 효율 최적화

- **Phase 2: 궤적 분석**
  - AIS 포인트를 선형 궤적(LineString)으로 변환
  - KD-Tree 알고리즘으로 근접 격자 검색 (100배 속도 향상)
  - Shapely Clip으로 궤적-격자 교차 계산
  - 격자별 통계: 선분 개수, 총 길이, 선박 정보
  - 일별 처리 및 결과 집계

- **결과 출력**
  - CSV 형식의 격자별 통계
  - Shapefile 형식의 GIS 데이터
  - 일별/전체 기간 통계

## 기술 스택

- Python 3.8+
- pandas - 데이터 처리
- geopandas - 지리공간 데이터 처리
- shapely - 기하학 연산
- scipy - KD-Tree 공간 인덱싱
- multiprocessing - 병렬 처리

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. Phase 1: 데이터 필터링

```bash
# 기본 필터링
python phase1_data_filtering.py

# 병렬 처리 버전 (더 빠름)
python phase1_parallel_filtering.py
```

**입력 파일:**
- th_ais_YYYYMMDD.csv - 원본 AIS 데이터
- 2024_tm_vessel_info.csv - 선박 정보

**출력 파일:**
- filtered_ais_data.csv - 필터링된 AIS 데이터

### 2. Phase 2: 궤적 분석

```bash
# KD-Tree + Clip 방식 (최신, 가장 빠름)
python phase2_KDTREE_CLIP_v1.py

# 다른 버전들
python phase2_kdtree_trajectory_v2.2.py
python phase2_point_based_trajectory.py
```

**입력 파일:**
- filtered_daily/*.csv - 일별 필터링된 데이터
- grid_polygon_wkt_new.csv - 격자 폴리곤 (WKT 형식)

**출력 파일:**
- results/daily_grids/grid_summary_YYYYMMDD.csv - 일별 격자 통계
- results/daily_shapefiles/grid_YYYYMMDD.shp - 일별 Shapefile

### 3. Shapefile 출력

```bash
python save_clipped_trajectories_to_shp.py
```

## 데이터 형식

### 입력: AIS 데이터
```
ship_id,latitude,longitude,sog,cog,heading,timestamp
<base64_encoded>,35.1234,129.3456,12.5,180.0,180,2024-12-01 00:00:00
```

### 출력: 격자 통계
```
grid_id,segment_count,total_length_km,ship_types,geometry
G001,150,45.6,"70,71,80",POLYGON((...))
```

## 프로젝트 구조

```
AIS_process/
├── phase1_data_filtering.py          # 기본 데이터 필터링
├── phase1_parallel_filtering.py      # 병렬 처리 필터링
├── phase2_KDTREE_CLIP_v1.py          # KD-Tree + Clip (최신)
├── phase2_kdtree_trajectory_v2.2.py  # KD-Tree 궤적 분석 v2.2
├── phase2_kdtree_trajectory_v2.1.py  # KD-Tree 궤적 분석 v2.1
├── phase2_kdtree_trajectory_v2.py    # KD-Tree 궤적 분석 v2
├── phase2_kdtree_trajectory_v1.py    # KD-Tree 궤적 분석 v1
├── phase2_kdtree_trajectory.py       # KD-Tree 궤적 분석 (초기)
├── phase2_point_based_trajectory.py  # 포인트 기반 처리
├── phase2_trajectory_analysis.py     # 궤적 분석
├── phase2_parallel_trajectory.py     # 병렬 궤적 처리
├── save_clipped_trajectories_to_shp.py  # Shapefile 출력
├── ais_trajectory_pipeline.py        # 전체 파이프라인
├── debug_phase1.py                   # 디버깅 도구
├── requirements.txt                  # 의존성 목록
└── .gitignore                        # Git 제외 파일 목록
```

## 성능 최적화

### KD-Tree 알고리즘
- 공간 인덱싱으로 O(log n) 검색
- 기존 전체 격자 탐색 대비 **100배 속도 향상**

### Clip 연산
- intersection() 대신 clip_by_rect() 사용
- 불필요한 계산 최소화

### 병렬 처리
- 멀티프로세싱으로 CPU 코어 활용
- 가용 메모리 기반 동적 프로세스 수 조정
- 청크 단위 처리로 메모리 사용량 최적화

### 메모리 관리
- 청크 단위 스트리밍 처리
- 불필요한 객체 명시적 해제 (gc.collect())
- 일별 분할 처리로 대용량 데이터 처리

## 처리 규모

- **입력 데이터:** 약 68GB (31일분 AIS 데이터)
- **처리 선박:** 화물선(70-79), 탱커선(80-89)
- **격자 수:** 수천~수만 개 격자
- **처리 시간:** 일별 약 10-30분 (시스템 사양에 따라 다름)

## 시스템 요구사항

- **CPU:** 멀티코어 프로세서 (8코어 이상 권장)
- **RAM:** 32GB 이상 권장 (16GB 최소)
- **저장공간:** 100GB 이상 (원본 데이터 + 결과 파일)
- **OS:** Linux (Ubuntu 18.04 테스트 완료)

## 주의사항

- 대용량 CSV 파일(.csv)은 Git 저장소에 포함되지 않습니다
- results/ 디렉토리의 출력 파일도 Git에서 제외됩니다
- 데이터 파일은 별도로 관리해야 합니다
- 처리 전 충분한 디스크 공간을 확보하세요

## 디버깅

```bash
# Phase 1 디버깅
python debug_phase1.py

# 로그 파일 확인
tail -f processing.log
tail -f phase2_progress.log
```

## 라이선스

이 프로젝트는 연구 및 분석 목적으로 개발되었습니다.

## 작성자

**Chaemin Yoon** (mccoals@naver.com)

## 버전 히스토리

- **v2.2** - KD-Tree + Clip 알고리즘 최적화
- **v2.1** - 메모리 관리 개선
- **v2.0** - KD-Tree 도입으로 성능 100배 향상
- **v1.0** - 초기 버전 (기본 intersection 방식)

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

---

**Last Updated:** 2024-11-06
