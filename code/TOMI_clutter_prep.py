"""
TOMI Clutter Data Preparation Pipeline
=======================================
Prepares terrain, building, and vegetation layers from open-source data
for integration into the TOMI CNN propagation model.

Steps:
  1. Define geographic grid from antenna CSV (matches TOMI's 1024×1024 grid)
  2. Terrain elevation from SRTM GeoTIFF
  3. Building footprints + heights from OpenStreetMap PBF
  4. Vegetation density from Sentinel-2 NDVI GeoTIFF
  5. Combine, normalize, and save all layers
  6. CNN integration helpers

Prerequisites:
  pip install rasterio numpy pandas pyrosm shapely

  For vegetation (Step 4), one of:
    pip install sentinelsat          (direct Sentinel-2 download)
    pip install earthengine-api      (Google Earth Engine)

  Data files needed:
    - Antenna CSV with 'Gda 94 Lat' and 'Gda 94 Long' columns
    - SRTM GeoTIFF (e.g., from OpenTopography or eio clip)
    - OpenStreetMap PBF extract for the region
    - (Optional) Sentinel-2 NDVI GeoTIFF
    - (Optional) GHSL Built-Up Height GeoTIFF for height gap-filling

Usage:
    python TOMI_clutter_prep.py \\
        --antenna-csv AntennasSA3_tilts.csv \\
        --srtm melbourne_srtm.tif \\
        --osm-pbf melbourne.osm.pbf \\
        --output-dir ./clutter_layers/ \\
        --ndvi melbourne_ndvi.tif \\
        --ghsl-height GHS_BUILT_H.tif
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════
# Pickle helpers (match TOMI's loadpickle / savepickle format)
# ══════════════════════════════════════════════════════════════════════

def savepickle(filename, obj):
    """Save array in TOMI's pickle format."""
    with open(filename, 'wb') as f:
        pickle.dump({"landscape": obj}, f)
    print(f"  Saved: {filename}  shape={obj.shape}  "
          f"range=[{obj.min():.4f}, {obj.max():.4f}]")


def loadpickle(filename):
    """Load array from TOMI's pickle format."""
    with open(filename, 'rb') as f:
        return pickle.load(f)["landscape"]


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Define the Geographic Grid
# ══════════════════════════════════════════════════════════════════════
#
# Reads the antenna CSV and extracts the bounding box.
# All clutter layers will be resampled to this exact grid.
#
# This matches the existing TOMI code:
#   MinAntennaLat  = df_antennas['Gda 94 Lat'].min() - 0.01
#   MaxAntennaLat  = df_antennas['Gda 94 Lat'].max() - 0.01
#   ...
#   a.y = int(1024*(row['Gda 94 Lat'] - MinAntennaLat)/(MaxAntennaLat - MinAntennaLat))
#   a.x = int(1024*(row['Gda 94 Long'] - MinAntennaLong)/(MaxAntennaLong - MinAntennaLong))

GRID_SIZE = 1024


def get_grid_bounds(antenna_csv_path):
    """
    Extract the geographic bounding box from the antenna CSV.

    Returns:
        bounds: (min_lon, min_lat, max_lon, max_lat) — same coordinate order as rasterio
        metadata: dict with lat/long ranges and cell size for logging
    """
    df = pd.read_csv(antenna_csv_path, low_memory=False)

    # Match TOMI's convention exactly
    min_lat = df['Gda 94 Lat'].min() - 0.01
    max_lat = df['Gda 94 Lat'].max() + 0.01  # Note: TOMI uses max - 0.01 as the upper
    min_lon = df['Gda 94 Long'].min() - 0.01
    max_lon = df['Gda 94 Long'].max() + 0.01

    # NOTE: The original TOMI code uses `max - 0.01` for BOTH min and max,
    # which is likely a copy-paste issue. Using min-0.01 and max+0.01 gives
    # a proper bounding box with margin. If you need exact TOMI compatibility,
    # change max_lat/max_lon to use -0.01 instead of +0.01.

    cell_width_deg = (max_lon - min_lon) / GRID_SIZE
    cell_height_deg = (max_lat - min_lat) / GRID_SIZE

    # Approximate cell size in meters (at Melbourne's latitude)
    cell_width_m = cell_width_deg * 111320 * np.cos(np.radians((min_lat + max_lat) / 2))
    cell_height_m = cell_height_deg * 110540

    metadata = {
        'min_lat': min_lat, 'max_lat': max_lat,
        'min_lon': min_lon, 'max_lon': max_lon,
        'cell_width_m': cell_width_m,
        'cell_height_m': cell_height_m,
        'n_antennas': len(df),
    }

    print("=" * 65)
    print("  STEP 1: Geographic Grid")
    print("=" * 65)
    print(f"  Antennas loaded: {len(df)}")
    print(f"  Lat range:  {min_lat:.5f} → {max_lat:.5f}")
    print(f"  Long range: {min_lon:.5f} → {max_lon:.5f}")
    print(f"  Grid:       {GRID_SIZE}×{GRID_SIZE}")
    print(f"  Cell size:  ~{cell_width_m:.1f}m × {cell_height_m:.1f}m")
    print()

    # bounds in (min_lon, min_lat, max_lon, max_lat) order for rasterio
    return (min_lon, min_lat, max_lon, max_lat), metadata


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Terrain Elevation from SRTM
# ══════════════════════════════════════════════════════════════════════
#
# Resamples the SRTM GeoTIFF to the TOMI grid.
# Output: 1024×1024 array of elevation in meters above sea level.
#
# What the CNN learns from this:
#   - Signals from hilltop antennas reach further downhill
#   - Ridges create shadow zones
#   - Valleys trap signals
#   - Tilt adjustments have asymmetric effects depending on terrain slope

def prepare_terrain_layer(srtm_path, bounds, grid_size=GRID_SIZE):
    """
    Resample SRTM elevation data to the TOMI grid.

    Args:
        srtm_path: path to SRTM GeoTIFF file
        bounds: (min_lon, min_lat, max_lon, max_lat)
        grid_size: output resolution (default 1024)

    Returns:
        terrain: numpy array shape (grid_size, grid_size), meters above sea level
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds

    print("=" * 65)
    print("  STEP 2: Terrain Elevation (SRTM)")
    print("=" * 65)

    min_lon, min_lat, max_lon, max_lat = bounds

    with rasterio.open(srtm_path) as src:
        print(f"  Source: {srtm_path}")
        print(f"  Source CRS: {src.crs}")
        print(f"  Source size: {src.width}×{src.height}")
        print(f"  Source bounds: {src.bounds}")

        # Target transform: maps pixel (col, row) → (lon, lat)
        target_transform = from_bounds(
            min_lon, min_lat, max_lon, max_lat,
            grid_size, grid_size)

        terrain = np.zeros((grid_size, grid_size), dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=terrain,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=src.crs,  # keep in same CRS (EPSG:4326)
            resampling=Resampling.bilinear)

    # Handle SRTM nodata values (voids are typically -32768)
    terrain[terrain < -100] = 0
    # Handle any NaN from reprojection
    terrain = np.nan_to_num(terrain, nan=0.0)

    print(f"  Output: {grid_size}×{grid_size}")
    print(f"  Elevation range: {terrain.min():.1f}m → {terrain.max():.1f}m")
    print(f"  Mean elevation: {terrain.mean():.1f}m")
    print()

    return terrain


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Building Footprints and Heights from OpenStreetMap
# ══════════════════════════════════════════════════════════════════════
#
# Produces two layers:
#   - Building density: fraction of each grid cell covered by buildings [0,1]
#   - Building height: average building height per cell in meters
#
# What the CNN learns:
#   - Dense building areas = high clutter loss
#   - Tall buildings between two antennas can block interference (beneficial)
#   - Tilt aimed at CBD produces different results than tilt aimed at suburbs

def prepare_building_layers(osm_pbf_path, bounds, grid_size=GRID_SIZE,
                            oversample=4):
    """
    Extract building density and height from OpenStreetMap.

    Args:
        osm_pbf_path: path to OSM PBF extract
        bounds: (min_lon, min_lat, max_lon, max_lat)
        grid_size: output resolution
        oversample: rasterize at this multiple of grid_size for density accuracy

    Returns:
        density: numpy array (grid_size, grid_size), values [0,1]
        height:  numpy array (grid_size, grid_size), meters
    """
    import pyrosm
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from rasterio import enums
    from shapely.geometry import mapping

    print("=" * 65)
    print("  STEP 3: Buildings from OpenStreetMap")
    print("=" * 65)

    min_lon, min_lat, max_lon, max_lat = bounds

    # ── Load buildings ──
    print(f"  Loading: {osm_pbf_path}")
    osm = pyrosm.OSM(osm_pbf_path, bounding_box=[min_lon, min_lat, max_lon, max_lat])
    buildings = osm.get_buildings()

    if buildings is None or len(buildings) == 0:
        print("  WARNING: No buildings found in OSM data!")
        print("  Returning zero layers.")
        return (np.zeros((grid_size, grid_size), dtype=np.float32),
                np.zeros((grid_size, grid_size), dtype=np.float32))

    print(f"  Buildings loaded: {len(buildings)}")

    # ── Reproject to EPSG:4326 if needed ──
    if buildings.crs and str(buildings.crs) != 'EPSG:4326':
        buildings = buildings.to_crs('EPSG:4326')
        print(f"  Reprojected to EPSG:4326")

    # ── Filter to bounding box ──
    # pyrosm's bounding_box may include buildings partially outside
    buildings = buildings.cx[min_lon:max_lon, min_lat:max_lat]
    print(f"  Buildings in grid bounds: {len(buildings)}")

    # ── Drop null geometries ──
    buildings = buildings[buildings.geometry.notna()]
    buildings = buildings[~buildings.geometry.is_empty]
    print(f"  Buildings with valid geometry: {len(buildings)}")

    if len(buildings) == 0:
        return (np.zeros((grid_size, grid_size), dtype=np.float32),
                np.zeros((grid_size, grid_size), dtype=np.float32))

    # ══════════════════════════════════════════════════════════════
    # 3a: Building Density
    # ══════════════════════════════════════════════════════════════
    # Rasterize at higher resolution, then downsample to get fractional
    # coverage. A 4x oversample means each grid cell aggregates 16 sub-pixels.

    print(f"  Computing density ({oversample}x oversample)...")

    hi_res = grid_size * oversample
    hi_transform = from_bounds(min_lon, min_lat, max_lon, max_lat,
                               hi_res, hi_res)

    # Rasterize: 1 inside buildings, 0 outside
    building_shapes = []
    for geom in buildings.geometry:
        try:
            building_shapes.append((mapping(geom), 1))
        except Exception:
            continue

    if len(building_shapes) == 0:
        return (np.zeros((grid_size, grid_size), dtype=np.float32),
                np.zeros((grid_size, grid_size), dtype=np.float32))

    building_mask = rasterize(
        building_shapes,
        out_shape=(hi_res, hi_res),
        transform=hi_transform,
        fill=0,
        dtype=np.uint8)

    # Downsample: average each oversample×oversample block → fraction [0,1]
    density = building_mask.reshape(
        grid_size, oversample, grid_size, oversample
    ).mean(axis=(1, 3)).astype(np.float32)

    print(f"  Density: mean={density.mean():.3f}, "
          f"max={density.max():.3f}, "
          f"nonzero cells={np.count_nonzero(density)}/{grid_size**2}")

    # ══════════════════════════════════════════════════════════════
    # 3b: Building Height
    # ══════════════════════════════════════════════════════════════
    # Parse height from OSM tags with fallback chain:
    #   1. 'height' tag (explicit meters)
    #   2. 'building:levels' × 3m per level
    #   3. Default 6m (2-story assumption)

    print("  Computing heights...")

    gdf = buildings.copy()

    # Parse explicit height tag
    gdf['height_m'] = np.nan
    if 'height' in gdf.columns:
        gdf['height_m'] = pd.to_numeric(
            gdf['height'].astype(str).str.replace('m', '').str.strip(),
            errors='coerce')

    # Fallback: building:levels × 3m
    if 'building:levels' in gdf.columns:
        levels = pd.to_numeric(gdf['building:levels'], errors='coerce')
        gdf['height_m'] = gdf['height_m'].fillna(levels * 3.0)

    # Default: 6m for anything with no tag
    gdf['height_m'] = gdf['height_m'].fillna(6.0)

    # Clamp unreasonable values
    gdf['height_m'] = gdf['height_m'].clip(lower=2.0, upper=350.0)

    height_tagged = gdf['height_m'].notna().sum()
    print(f"  Height sources: {height_tagged} buildings "
          f"({100*height_tagged/len(gdf):.0f}% with explicit/level tags)")

    # Rasterize at grid resolution (max height per cell)
    target_transform = from_bounds(min_lon, min_lat, max_lon, max_lat,
                                   grid_size, grid_size)

    height_shapes = []
    for _, row in gdf.iterrows():
        if row.geometry is not None and not row.geometry.is_empty:
            try:
                height_shapes.append((mapping(row.geometry), float(row.height_m)))
            except Exception:
                continue

    if len(height_shapes) > 0:
        height = rasterize(
            height_shapes,
            out_shape=(grid_size, grid_size),
            transform=target_transform,
            fill=0,
            dtype=np.float32,
            merge_alg=enums.MergeAlg.replace)
    else:
        height = np.zeros((grid_size, grid_size), dtype=np.float32)

    print(f"  Height: mean (where >0)={height[height>0].mean():.1f}m, "
          f"max={height.max():.1f}m")
    print()

    return density, height


def gapfill_height_with_ghsl(osm_height, ghsl_path, bounds, grid_size=GRID_SIZE):
    """
    Fill missing building heights with GHSL Built-Up Height layer.

    GHSL provides satellite-derived building height estimates at 10m
    resolution globally. Free from ghsl.jrc.ec.europa.eu

    Use OSM height where available (more accurate for individual buildings),
    fall back to GHSL where OSM has no data.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds

    print("  Gap-filling heights with GHSL...")

    min_lon, min_lat, max_lon, max_lat = bounds

    with rasterio.open(ghsl_path) as src:
        target_transform = from_bounds(
            min_lon, min_lat, max_lon, max_lat,
            grid_size, grid_size)

        ghsl_height = np.zeros((grid_size, grid_size), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=ghsl_height,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs='EPSG:4326',
            resampling=Resampling.bilinear)

    ghsl_height = np.nan_to_num(ghsl_height, nan=0.0)
    ghsl_height[ghsl_height < 0] = 0

    # Merge: OSM where available, GHSL where OSM is zero
    combined = np.where(osm_height > 0, osm_height, ghsl_height)

    osm_cells = np.count_nonzero(osm_height)
    ghsl_filled = np.count_nonzero(combined) - osm_cells
    print(f"  OSM height cells: {osm_cells}, GHSL gap-filled: {ghsl_filled}")
    print()

    return combined


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Vegetation Density from Sentinel-2 NDVI
# ══════════════════════════════════════════════════════════════════════
#
# If an NDVI GeoTIFF is provided, resample it to the grid.
# If not provided, generate a proxy from OpenStreetMap landuse tags.
#
# NDVI values:
#   < 0.1  : water, bare ground, urban surfaces
#   0.1–0.3: sparse vegetation, grassland
#   0.3–0.6: moderate vegetation, shrubs, scattered trees
#   > 0.6  : dense canopy, forest
#
# What the CNN learns:
#   - Dense tree canopy adds 10–20 dB loss at 1800 MHz
#   - Tilt toward parkland behaves differently than toward concrete
#   - Suburban streets with tree cover have more loss than bare streets

def prepare_vegetation_layer(ndvi_tif_path, bounds, grid_size=GRID_SIZE):
    """
    Resample Sentinel-2 NDVI GeoTIFF to the TOMI grid.

    Args:
        ndvi_tif_path: path to NDVI GeoTIFF
        bounds: (min_lon, min_lat, max_lon, max_lat)

    Returns:
        ndvi: numpy array (grid_size, grid_size), values [0,1]
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds

    print("=" * 65)
    print("  STEP 4: Vegetation (Sentinel-2 NDVI)")
    print("=" * 65)
    print(f"  Source: {ndvi_tif_path}")

    min_lon, min_lat, max_lon, max_lat = bounds

    with rasterio.open(ndvi_tif_path) as src:
        print(f"  Source CRS: {src.crs}")
        print(f"  Source size: {src.width}×{src.height}")

        target_transform = from_bounds(
            min_lon, min_lat, max_lon, max_lat,
            grid_size, grid_size)

        ndvi = np.zeros((grid_size, grid_size), dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=ndvi,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs='EPSG:4326',
            resampling=Resampling.bilinear)

    ndvi = np.nan_to_num(ndvi, nan=0.0)

    # Clamp to [0, 1] — negative NDVI (water) → 0
    ndvi = np.clip(ndvi, 0.0, 1.0)

    print(f"  Output: {grid_size}×{grid_size}")
    print(f"  NDVI range: {ndvi.min():.3f} → {ndvi.max():.3f}")
    print(f"  Dense vegetation (>0.6): "
          f"{100 * (ndvi > 0.6).sum() / ndvi.size:.1f}% of cells")
    print()

    return ndvi


def prepare_vegetation_proxy_from_osm(osm_pbf_path, bounds, grid_size=GRID_SIZE):
    """
    Fallback: estimate vegetation from OSM landuse tags if no NDVI available.

    Maps OSM landuse/natural tags to approximate NDVI values:
      forest/wood       → 0.7
      park/garden       → 0.5
      grass/meadow      → 0.3
      farmland          → 0.25
      scrub             → 0.4
      water             → 0.0
      residential       → 0.15  (some garden vegetation)
      commercial/retail → 0.05  (minimal vegetation)
      industrial        → 0.02
    """
    import pyrosm
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping

    print("=" * 65)
    print("  STEP 4: Vegetation (OSM proxy — no NDVI file provided)")
    print("=" * 65)

    min_lon, min_lat, max_lon, max_lat = bounds

    osm = pyrosm.OSM(osm_pbf_path, bounding_box=[min_lon, min_lat, max_lon, max_lat])

    # Try to get landuse and natural features
    landuse = osm.get_landuse()
    natural = osm.get_natural()

    ndvi_map = {
        # landuse tags
        'forest': 0.7, 'wood': 0.7,
        'orchard': 0.5,
        'recreation_ground': 0.4, 'park': 0.5, 'garden': 0.5,
        'allotments': 0.4,
        'grass': 0.3, 'meadow': 0.3, 'village_green': 0.3,
        'farmland': 0.25, 'farmyard': 0.2,
        'vineyard': 0.35,
        'scrub': 0.4, 'heath': 0.35,
        'residential': 0.15,
        'commercial': 0.05, 'retail': 0.05,
        'industrial': 0.02,
        # natural tags
        'wood': 0.7, 'tree_row': 0.5,
        'water': 0.0, 'wetland': 0.3,
        'bare_rock': 0.0, 'sand': 0.0,
    }

    shapes = []

    for gdf_source, tag_col in [(landuse, 'landuse'), (natural, 'natural')]:
        if gdf_source is None or len(gdf_source) == 0:
            continue
        if gdf_source.crs and str(gdf_source.crs) != 'EPSG:4326':
            gdf_source = gdf_source.to_crs('EPSG:4326')
        for _, row in gdf_source.iterrows():
            tag_val = str(row.get(tag_col, '')).lower()
            ndvi_val = ndvi_map.get(tag_val, 0.1)
            if row.geometry is not None and not row.geometry.is_empty:
                try:
                    shapes.append((mapping(row.geometry), ndvi_val))
                except Exception:
                    continue

    target_transform = from_bounds(min_lon, min_lat, max_lon, max_lat,
                                   grid_size, grid_size)

    if len(shapes) > 0:
        vegetation = rasterize(
            shapes,
            out_shape=(grid_size, grid_size),
            transform=target_transform,
            fill=0.1,  # default: sparse background vegetation
            dtype=np.float32)
    else:
        print("  WARNING: No landuse features found. Using uniform 0.1.")
        vegetation = np.full((grid_size, grid_size), 0.1, dtype=np.float32)

    vegetation = np.clip(vegetation, 0.0, 1.0)
    print(f"  OSM landuse/natural features used: {len(shapes)}")
    print(f"  Proxy NDVI range: {vegetation.min():.3f} → {vegetation.max():.3f}")
    print()

    return vegetation


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Combine, Normalize, and Save
# ══════════════════════════════════════════════════════════════════════

def normalize_layer(layer, method='minmax', cap=None):
    """
    Normalize a layer to [0, 1].

    Args:
        layer: numpy array
        method: 'minmax' or 'cap'
        cap: if method='cap', divide by this value and clip to [0,1]
    """
    if method == 'cap' and cap is not None:
        return np.clip(layer / cap, 0.0, 1.0)
    else:
        vmin, vmax = layer.min(), layer.max()
        if vmax - vmin < 1e-8:
            return np.zeros_like(layer)
        return (layer - vmin) / (vmax - vmin)


def save_all_layers(output_dir, terrain_raw, building_density, building_height,
                    vegetation):
    """
    Normalize all layers and save in TOMI pickle format.

    Saves both normalized (for CNN input) and raw (for diagnostics/LOS checks).
    """
    print("=" * 65)
    print("  STEP 5: Normalize and Save")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)

    # ── Terrain: min-max normalize ──
    terrain_norm = normalize_layer(terrain_raw, method='minmax')

    # ── Building density: already [0, 1] ──
    density_norm = building_density

    # ── Building height: cap at 100m ──
    # Melbourne's tallest is ~300m (Australia 108) but 99% of buildings
    # are under 100m. Capping at 100 preserves detail in the common range.
    height_norm = normalize_layer(building_height, method='cap', cap=100.0)

    # ── Vegetation: already [0, 1] ──
    veg_norm = vegetation

    # ── Save normalized layers (for CNN) ──
    savepickle(os.path.join(output_dir, 'clutter_terrain.pkl'), terrain_norm)
    savepickle(os.path.join(output_dir, 'clutter_building_density.pkl'), density_norm)
    savepickle(os.path.join(output_dir, 'clutter_building_height.pkl'), height_norm)
    savepickle(os.path.join(output_dir, 'clutter_vegetation.pkl'), veg_norm)

    # ── Save raw terrain (for line-of-sight checks in propagation model) ──
    savepickle(os.path.join(output_dir, 'terrain_raw_meters.pkl'), terrain_raw)

    # ── Save a combined 4-channel numpy array (convenience for CNN loading) ──
    combined = np.stack([terrain_norm, density_norm, height_norm, veg_norm])
    np.save(os.path.join(output_dir, 'clutter_4ch.npy'), combined)
    print(f"  Saved: {os.path.join(output_dir, 'clutter_4ch.npy')}  "
          f"shape={combined.shape}")

    # ── Summary ──
    print()
    print("  Layer Summary:")
    print(f"  {'Layer':<25} {'Min':>8} {'Max':>8} {'Mean':>8} {'Nonzero%':>10}")
    print("  " + "-" * 60)
    for name, layer in [('Terrain (norm)', terrain_norm),
                         ('Building Density', density_norm),
                         ('Building Height (norm)', height_norm),
                         ('Vegetation', veg_norm)]:
        nz = 100 * np.count_nonzero(layer) / layer.size
        print(f"  {name:<25} {layer.min():>8.3f} {layer.max():>8.3f} "
              f"{layer.mean():>8.3f} {nz:>9.1f}%")
    print()

    return terrain_norm, density_norm, height_norm, veg_norm


# ══════════════════════════════════════════════════════════════════════
# STEP 6: CNN Integration Helpers
# ══════════════════════════════════════════════════════════════════════
#
# These functions show how to load the clutter layers and integrate
# them into the TOMI CNN training loop.

def load_clutter_for_cnn(clutter_dir, device='cpu'):
    """
    Load clutter layers and prepare a tensor for CNN input.

    Returns:
        clutter_tensor: shape (1, 4, 1024, 1024), double precision
                        channels: [terrain, building_density, building_height, vegetation]

    Usage in training loop:
        clutter = load_clutter_for_cnn('./clutter_layers/', device)

        # For each state (shape 1,8,1024,1024), prepend clutter:
        state_with_clutter = torch.cat([state, clutter], dim=1)  # → (1,12,1024,1024)
        output = model(state_with_clutter)
    """
    import torch

    combined = np.load(os.path.join(clutter_dir, 'clutter_4ch.npy'))  # (4, 1024, 1024)

    clutter_tensor = torch.from_numpy(combined).unsqueeze(0).double().to(device)
    # shape: (1, 4, 1024, 1024)

    print(f"  Clutter tensor loaded: shape={tuple(clutter_tensor.shape)}, "
          f"device={clutter_tensor.device}")

    return clutter_tensor


def get_modified_cnn_code():
    """
    Print the CNN modifications needed to accept clutter channels.
    """
    print("""
    ══════════════════════════════════════════════════════════════
    CNN Modifications for Clutter Integration
    ══════════════════════════════════════════════════════════════

    1. Change the first Conv2d input channels from 8 to 12:

       # ORIGINAL:
       self.c0 = nn.Conv2d(8, 16, 8)

       # NEW (8 environment + 4 clutter channels):
       self.c0 = nn.Conv2d(12, 16, 8)

    2. Load clutter once at startup:

       clutter = load_clutter_for_cnn('./clutter_layers/', device)

    3. In the training loop, concatenate clutter to each state:

       # state shape: (batch, 8, 1024, 1024)
       # clutter shape: (1, 4, 1024, 1024) — broadcast over batch

       state_with_clutter = torch.cat([
           state,
           clutter.expand(state.size(0), -1, -1, -1)
       ], dim=1)
       # → shape: (batch, 12, 1024, 1024)

       output = model(state_with_clutter)

    4. Same for next_state in target computation:

       state_1_with_clutter = torch.cat([
           state_1_batch,
           clutter.expand(state_1_batch.size(0), -1, -1, -1)
       ], dim=1)

       target_q_next = target_model(state_1_with_clutter)

    ══════════════════════════════════════════════════════════════
    """)


# ══════════════════════════════════════════════════════════════════════
# MAIN: Full Pipeline
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='TOMI Clutter Data Preparation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with all data sources:
  python TOMI_clutter_prep.py \\
      --antenna-csv AntennasSA3_tilts.csv \\
      --srtm melbourne_srtm.tif \\
      --osm-pbf melbourne.osm.pbf \\
      --ndvi melbourne_ndvi.tif \\
      --ghsl-height GHS_BUILT_H.tif \\
      --output-dir ./clutter_layers/

  # Minimum viable (no NDVI, no GHSL — uses OSM proxy for vegetation):
  python TOMI_clutter_prep.py \\
      --antenna-csv AntennasSA3_tilts.csv \\
      --srtm melbourne_srtm.tif \\
      --osm-pbf melbourne.osm.pbf \\
      --output-dir ./clutter_layers/

  # Print CNN integration instructions only:
  python TOMI_clutter_prep.py --show-cnn-code
        """)

    parser.add_argument('--antenna-csv', type=str,
                        help='Path to antenna CSV with Gda 94 Lat/Long columns')
    parser.add_argument('--srtm', type=str,
                        help='Path to SRTM elevation GeoTIFF')
    parser.add_argument('--osm-pbf', type=str,
                        help='Path to OpenStreetMap PBF extract')
    parser.add_argument('--ndvi', type=str, default=None,
                        help='Path to Sentinel-2 NDVI GeoTIFF (optional, uses OSM proxy if missing)')
    parser.add_argument('--ghsl-height', type=str, default=None,
                        help='Path to GHSL Built-Up Height GeoTIFF (optional, for gap-filling)')
    parser.add_argument('--output-dir', type=str, default='./clutter_layers/',
                        help='Output directory for clutter pickle files')
    parser.add_argument('--show-cnn-code', action='store_true',
                        help='Print CNN integration code and exit')

    args = parser.parse_args()

    if args.show_cnn_code:
        get_modified_cnn_code()
        return

    if not args.antenna_csv or not args.srtm or not args.osm_pbf:
        parser.error("--antenna-csv, --srtm, and --osm-pbf are required")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         TOMI Clutter Data Preparation Pipeline             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── STEP 1: Geographic grid ──
    bounds, metadata = get_grid_bounds(args.antenna_csv)

    # ── STEP 2: Terrain ──
    terrain_raw = prepare_terrain_layer(args.srtm, bounds)

    # ── STEP 3: Buildings ──
    building_density, building_height = prepare_building_layers(
        args.osm_pbf, bounds)

    # Gap-fill height with GHSL if provided
    if args.ghsl_height and os.path.isfile(args.ghsl_height):
        building_height = gapfill_height_with_ghsl(
            building_height, args.ghsl_height, bounds)

    # ── STEP 4: Vegetation ──
    if args.ndvi and os.path.isfile(args.ndvi):
        vegetation = prepare_vegetation_layer(args.ndvi, bounds)
    else:
        print("  No NDVI file provided. Using OSM landuse as proxy.")
        vegetation = prepare_vegetation_proxy_from_osm(args.osm_pbf, bounds)

    # ── STEP 5: Normalize and save ──
    save_all_layers(args.output_dir,
                    terrain_raw, building_density, building_height, vegetation)

    # ── STEP 6: Show CNN integration ──
    get_modified_cnn_code()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                      Pipeline Complete                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Output directory: {args.output_dir}")
    print(f"  Files created:")
    for f in ['clutter_terrain.pkl', 'clutter_building_density.pkl',
              'clutter_building_height.pkl', 'clutter_vegetation.pkl',
              'terrain_raw_meters.pkl', 'clutter_4ch.npy']:
        filepath = os.path.join(args.output_dir, f)
        if os.path.isfile(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"    {f:<40} {size_mb:.1f} MB")
    print()
    print("  To use in TOMI Q-learner:")
    print(f"    clutter = load_clutter_for_cnn('{args.output_dir}', device)")
    print()


if __name__ == '__main__':
    main()
