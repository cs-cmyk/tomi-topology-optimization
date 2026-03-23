"""
TOMI Q-Learner
==============
Reinforcement learning framework for LTE antenna tilt and azimuth optimization.

Architecture:
  - Dueling Double DQN with N-step returns and target network
  - Adaptive SINR/RSRQ reward with graduated overlap penalty
  - Separate electrical/mechanical tilt (only e-tilt is actionable via RET)
  - Cached footprint computation with incremental layer updates
  - Simulated annealing local refinement
  - PPM validation pipeline with small-sample Bayesian testing
  - Mixed precision: fp32 weights, fp16 forward (autocast), fp64 CuPy grids

CNN: 4-block progressive downsample (receptive field ~400px), dueling V(s)+A(s,a),
     separate cached clutter encoder. ~3M parameters.

Propagation: Local-patch make_norm (~1.2ms vs ~5ms full grid), batch init via
     make_norm_batch (~20x faster), FootprintCache for incremental per-step updates.
"""

import cupy as np
import numpy as npx
from matplotlib import cm
from PIL import Image
from scipy.spatial import distance
import pandas as pd
import random
import os
import pickle
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from collections import deque


# ══════════════════════════════════════════════════════════════════════
# Globals
# ══════════════════════════════════════════════════════════════════════

max_w, max_l = 1024, 1024
landscape = npx.zeros((max_w, max_l))

NOISE_FLOOR = 1e-10       # for SINR/RSRQ computation (linear scale)
N_RB = 50                 # resource blocks, typical for 10 MHz LTE
N_STEP = 3                # N-step return horizon


# ══════════════════════════════════════════════════════════════════════
# Antenna
# ══════════════════════════════════════════════════════════════════════
# e_tilt: electrical tilt, remotely adjustable via RET [0, -10 deg]
# m_tilt: mechanical tilt, fixed (set by physical mounting)
# r:      effective tilt = e_tilt + m_tilt (used by propagation model)
# Only e_tilt is in the optimizer's action space.

class Antenna:
    def __init__(self):
        self.x = np.random.randint(0, max_w)
        self.y = np.random.randint(0, max_l)
        self.z = 1
        self.e_tilt = 0
        self.m_tilt = 0
        self.r = 0
        self.r1 = np.random.randint(-90, 10) % 360   # azimuth
        self.r2 = np.random.randint(10, 360)
        self.tech = 0

    def update_effective_tilt(self):
        """Recompute effective tilt from components. Call after modifying e_tilt."""
        self.r = self.e_tilt + self.m_tilt
        self.r2 = 360 + self.r

G_antennas = []


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def makeImage(land):
    colmap = cm.get_cmap('gist_stern')
    lut = npx.array([colmap(i / 255)[:3] for i in range(256)]) * 255
    grey = (land * 255 / land.max()).astype(int)
    result = npx.zeros((*grey.shape, 3), dtype=np.uint8)
    npx.take(lut, grey, axis=0, out=result)
    return Image.fromarray(result)


def make_norm(ant, grid_size=1024):
    """Compute antenna coverage footprint using local-patch optimization.
    Only computes trig within the estimated bounding box of the beam,
    then writes into a full-grid array. ~4-8x faster than full-grid."""
    r_min, r_max, c_min, c_max = _footprint_bbox(ant, grid_size)

    h = r_max - r_min
    w = c_max - c_min
    if h <= 0 or w <= 0:
        return np.zeros((grid_size, grid_size))

    # Local coordinate grids (much smaller than full 1024x1024)
    cols = np.arange(c_min, c_max).reshape(1, -1)
    rows = np.arange(r_min, r_max).reshape(-1, 1)

    local_dx = cols - ant.x
    local_dy = rows - ant.y
    local_dz = np.full((h, w), -ant.z, dtype=np.float64)

    # 3D direction vector from tilt and azimuth
    r2 = (360 + ant.r) * np.pi / 180
    r1 = ant.r1 * np.pi / 180
    direction = [np.cos(r2) * np.cos(r1), np.cos(r2) * np.sin(r1), np.sin(r2)]

    dist = np.sqrt(local_dx**2 + local_dy**2 + local_dz**2)
    dot = local_dx * direction[0] + local_dy * direction[1] + local_dz * direction[2]
    recip = np.reciprocal(dist)
    angle = np.abs(np.arccos(np.clip(dot * recip, -1.0, 1.0))) * (180 / np.pi)

    local_vals = (50 * np.pi / 180) * recip
    local_vals[angle > 22] = 0

    result = np.zeros((grid_size, grid_size))
    result[r_min:r_max, c_min:c_max] = local_vals
    return result


def _footprint_bbox(ant, grid_size=1024, max_range_px=500):
    """Estimate bounding box of an antenna's coverage wedge."""
    az_rad = ant.r1 * np.pi / 180
    dx = float(np.cos(az_rad))
    dy = float(np.sin(az_rad))

    lateral = int(max_range_px * 0.45)
    cx = ant.x + dx * max_range_px * 0.5
    cy = ant.y + dy * max_range_px * 0.5
    half = max(max_range_px // 2 + lateral, lateral + 50)

    return (max(0, int(cy - half)), min(grid_size, int(cy + half)),
            max(0, int(cx - half)), min(grid_size, int(cx + half)))


def make_norm_batch(antennas, grid_size=1024, chunk_size=32):
    """Batch-compute footprints for multiple antennas in parallel.
    CuPy parallelizes trig across the chunk dimension on GPU.
    ~8-16x faster than sequential for initial environment build.
    Memory: chunk_size x 1024^2 x 8 bytes x ~5 intermediates."""
    N = len(antennas)
    all_footprints = []

    cols = np.arange(grid_size).reshape(1, 1, grid_size)
    rows = np.arange(grid_size).reshape(1, grid_size, 1)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        batch = antennas[start:end]
        B = len(batch)

        ax = np.array([a.x for a in batch]).reshape(B, 1, 1)
        ay = np.array([a.y for a in batch]).reshape(B, 1, 1)
        az = np.array([a.z for a in batch]).reshape(B, 1, 1)
        r2 = np.array([360 + a.r for a in batch]).reshape(B, 1, 1) * np.pi / 180
        r1 = np.array([a.r1 for a in batch]).reshape(B, 1, 1) * np.pi / 180

        dir_x = np.cos(r2) * np.cos(r1)
        dir_y = np.cos(r2) * np.sin(r1)
        dir_z = np.sin(r2)

        dx = cols - ax
        dy = rows - ay
        dz = -az

        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dot = dx * dir_x + dy * dir_y + dz * dir_z
        recip = np.reciprocal(dist)
        angle = np.abs(np.arccos(np.clip(dot * recip, -1.0, 1.0))) * (180 / np.pi)

        vals = (50 * np.pi / 180) * recip
        vals[angle > 22] = 0
        vals[vals < 0.01] = 0

        for i in range(B):
            all_footprints.append(vals[i])

        del dx, dy, dz, dist, dot, recip, angle, vals

    return all_footprints


def loadpickle(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)["landscape"]


def savepickle(filename, obj):
    with open(filename, 'wb') as outfile:
        pickle.dump({"landscape": obj}, outfile)


# ══════════════════════════════════════════════════════════════════════
# Antenna Loading
# ══════════════════════════════════════════════════════════════════════

def load_antennas(filename_ant):
    """Load antennas from CSV with separate electrical and mechanical tilt."""
    df_antennas = pd.read_csv(filename_ant, low_memory=False)

    MinAntennaLat = df_antennas['Gda 94 Lat'].min() - 0.01
    MaxAntennaLat = df_antennas['Gda 94 Lat'].max() - 0.01
    MinAntennaLong = df_antennas['Gda 94 Long'].min() - 0.01
    MaxAntennaLong = df_antennas['Gda 94 Long'].max() - 0.01

    df_antennas['E Tilt'] = df_antennas['E Tilt'].str.replace('', '0')
    df_antennas['E Tilt'] = df_antennas['E Tilt'].str.replace('*', '0')
    df_antennas['E Tilt'] = df_antennas['E Tilt'].str.replace(',', '')
    df_antennas = df_antennas.dropna()
    df_antennas['E Tilt'] = df_antennas['E Tilt'].astype(int)

    antennas_out = []
    for index, row in df_antennas.iterrows():
        a = Antenna()
        a.y = int(1024 * (row['Gda 94 Lat'] - MinAntennaLat) / (MaxAntennaLat - MinAntennaLat))
        a.x = int(1024 * (row['Gda 94 Long'] - MinAntennaLong) / (MaxAntennaLong - MinAntennaLong))
        a.z1 = row['Ant. Height']
        a.z = row['Ant. Height']
        a.r1 = row['Azimuth']
        a.tech = 18
        a.m_tilt = -row['M Tilt']
        a.e_tilt = max(min(-row['E Tilt'], 0), -10)
        a.update_effective_tilt()
        antennas_out.append(a)

    return antennas_out


# ══════════════════════════════════════════════════════════════════════
# Footprint Cache
# ══════════════════════════════════════════════════════════════════════
# Caches per-antenna coverage footprints. At init, batch-computes all N
# footprints. Per training step, recomputes only the one antenna that
# changed. Maintains serving, total, and overlap layers incrementally.

class FootprintCache:

    def __init__(self, antennas, grid_size=1024):
        self.grid_size = grid_size
        self.n_antennas = len(antennas)

        # Batch-compute all footprints at init (~20x faster than sequential)
        print(f"Building footprint cache for {self.n_antennas} antennas...")
        self.footprints = make_norm_batch(antennas, grid_size)
        for i in range(len(self.footprints)):
            self.footprints[i][self.footprints[i] < 0.01] = 0

        # Build layers from cached footprints
        self.total_layer = np.zeros((grid_size, grid_size))
        self.serving_layer = np.zeros((grid_size, grid_size))
        self.overlap_layer = np.zeros((grid_size, grid_size), dtype=int)

        for fp in self.footprints:
            self.total_layer += fp
            self.serving_layer = np.maximum(self.serving_layer, fp)
            self.overlap_layer[fp > 0] += 1

        print(f"  Cache built. Serving range: "
              f"[{float(self.serving_layer.min()):.4f}, {float(self.serving_layer.max()):.4f}]")

    def update_antenna(self, idx, new_ant):
        """Recompute one antenna's footprint and update layers incrementally.
        This is the hot path — called once per training step."""
        old_fp = self.footprints[idx]

        # Remove old footprint
        self.total_layer -= old_fp
        self.total_layer[self.total_layer < 0] = 0
        self.overlap_layer[old_fp > 0] -= 1
        self.overlap_layer[self.overlap_layer < 0] = 0

        # Compute new footprint (local-patch, ~1.2ms)
        new_fp = make_norm(new_ant, self.grid_size)
        new_fp[new_fp < 0.01] = 0

        # Add new footprint
        self.total_layer += new_fp
        self.overlap_layer[new_fp > 0] += 1

        # Update cache
        self.footprints[idx] = new_fp

        # Rebuild serving layer only in the affected region
        affected = (old_fp > 0) | (new_fp > 0)
        if affected.any():
            self.serving_layer[affected] = 0
            for fp in self.footprints:
                self.serving_layer[affected] = np.maximum(
                    self.serving_layer[affected], fp[affected])


# ══════════════════════════════════════════════════════════════════════
# Adaptive SINR/RSRQ Reward with Graduated Overlap Penalty
# ══════════════════════════════════════════════════════════════════════
# R = R_quality * R_overlap - R_coverage
#
# R_quality:  alpha-weighted blend of SINR and RSRQ, population-weighted
#             alpha -> 1 (urban, interference-limited): lean on SINR
#             alpha -> 0 (rural, noise-limited): lean on RSRQ
# R_overlap:  graduated scoring (0->0.5, 1->1.0, 2->1.2, 3+->decreasing)
# R_coverage: squared penalty for locations below min serving power

def compute_adaptive_reward(serving_layer, total_layer, population_env, overlap_layer,
                            min_power=0.005):
    """Compute combined reward: R_quality + R_overlap + R_coverage."""
    interference = total_layer - serving_layer
    interference[interference < 0] = 0

    sinr = serving_layer / (interference + NOISE_FLOOR)
    rsrq = N_RB * serving_layer / (total_layer + NOISE_FLOOR)

    alpha = interference / (interference + NOISE_FLOOR)

    combined_quality = alpha * np.log2(1 + sinr) + (1 - alpha) * np.log2(1 + rsrq)

    overlap_score = np.where(overlap_layer == 0, 0.5,
                   np.where(overlap_layer == 1, 1.0,
                   np.where(overlap_layer == 2, 1.2,
                            1.2 - 0.3 * (overlap_layer - 2))))
    overlap_score = np.clip(overlap_score, 0.0, 1.5)

    reward = np.sum(combined_quality * population_env * overlap_score)

    coverage_gap = np.maximum(0, min_power - serving_layer)
    reward -= np.sum(coverage_gap ** 2 * population_env)

    return reward


# ══════════════════════════════════════════════════════════════════════
# Action Space
# ══════════════════════════════════════════════════════════════════════
# 8 actions per antenna: fine/coarse tilt and azimuth.
# Only e_tilt is modified (RET-adjustable). m_tilt is never touched.

ACTION_DEFS = {
    0: ('e_tilt', -0.5, -10.0, 0.0),    # fine downtilt
    1: ('e_tilt', +0.5, -10.0, 0.0),    # fine uptilt
    2: ('e_tilt', -2.0, -10.0, 0.0),    # coarse downtilt
    3: ('e_tilt', +2.0, -10.0, 0.0),    # coarse uptilt
    4: ('r1',    -1.0,   0.0, 359.0),   # azimuth left 1 deg
    5: ('r1',    +1.0,   0.0, 359.0),   # azimuth right 1 deg
    6: ('r1',    -5.0,   0.0, 359.0),   # azimuth left 5 deg
    7: ('r1',    +5.0,   0.0, 359.0),   # azimuth right 5 deg
}


def frame_step(idx, antennas, antennas_old, cache, population_environment):
    """Execute one action using cached footprints.
    Only recomputes the single antenna that changed (~3ms total)."""
    n_actions_per_ant = len(ACTION_DEFS)
    antenna_idx = idx // n_actions_per_ant
    action_subidx = idx % n_actions_per_ant

    ant = antennas[antenna_idx]

    ant1 = Antenna()
    ant1.x, ant1.y, ant1.z = ant.x, ant.y, ant.z
    ant1.e_tilt, ant1.m_tilt, ant1.r = ant.e_tilt, ant.m_tilt, ant.r
    ant1.r1, ant1.r2 = ant.r1, ant.r2

    param, delta, min_bound, max_bound = ACTION_DEFS[action_subidx]
    new_val = getattr(ant1, param) + delta

    if param == 'r1':
        new_val = new_val % 360

    if param == 'e_tilt' and (new_val < min_bound or new_val > max_bound):
        return cache.serving_layer, 0, 1

    setattr(ant1, param, new_val)
    if param == 'e_tilt':
        ant1.update_effective_tilt()

    reward_old = compute_adaptive_reward(
        cache.serving_layer, cache.total_layer,
        population_environment, cache.overlap_layer)

    # Incremental update: recompute only this antenna's footprint
    cache.update_antenna(antenna_idx, ant1)

    reward_new = compute_adaptive_reward(
        cache.serving_layer, cache.total_layer,
        population_environment, cache.overlap_layer)

    reward = (reward_new - reward_old) / abs(reward_old) if abs(reward_old) > 1e-12 else 0.0

    antennas_old[antenna_idx] = ant
    antennas[antenna_idx] = ant1

    return cache.serving_layer, reward, 0


# ══════════════════════════════════════════════════════════════════════
# Dueling CNN Q-Network
# ══════════════════════════════════════════════════════════════════════
# 4-block progressive downsampling: 1024->256->64->16->4
# Receptive field ~400px (captures inter-antenna interference).
# Separate cached clutter encoder for static terrain/building/vegetation.
# Dueling head: Q(s,a) = V(s) + A(s,a) - mean(A)
# No output activation (Q-values can be negative).

class PropogatorCNN2(nn.Module):

    def __init__(self, max_antennas=100, use_clutter=False):
        super().__init__()

        self.n_antennas = max_antennas
        self.n_actions_per_antenna = len(ACTION_DEFS)
        self.n_actions = self.n_actions_per_antenna * self.n_antennas
        self.use_clutter = use_clutter

        self.experience_size = 50000
        self.end_epsilon = 0.0001
        self.start_epsilon = 0.5
        self.gamma = 0.99
        self.n_iters = 100000
        self.mini_batch_size = 8
        self.target_update_freq = 500
        self.tau = 0.005

        # Dynamic encoder: 8-ch environment state (changes every step)
        self.env_encoder = nn.Sequential(
            nn.Conv2d(8, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(4, stride=4),          # 1024 -> 256
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(4, stride=4),          # 256 -> 64
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(4, stride=4),          # 64 -> 16
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(4, stride=4),          # 16 -> 4
        )   # -> (batch, 128, 4, 4) = 2048 features

        # Static clutter encoder (optional): precomputed once at startup
        self.clutter_encoder = None
        self._clutter_features = None

        if use_clutter:
            self.clutter_encoder = nn.Sequential(
                nn.Conv2d(4, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(4, stride=4),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(4, stride=4),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )   # -> (batch, 64, 4, 4) = 1024 features

        # Dueling head: V(s) + A(s,a) - mean(A)
        flatten_size = 128 * 4 * 4
        if use_clutter:
            flatten_size += 64 * 4 * 4

        self.value_stream = nn.Sequential(
            nn.Linear(flatten_size, 256), nn.ReLU(), nn.Linear(256, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(flatten_size, 256), nn.ReLU(), nn.Linear(256, self.n_actions))

    def precompute_clutter(self, clutter_tensor):
        """Encode clutter once and cache. Call at startup with shape (1, 4, 1024, 1024)."""
        if self.clutter_encoder is not None:
            with torch.no_grad():
                self._clutter_features = self.clutter_encoder(clutter_tensor)

    def forward(self, x):
        """x: (batch, 8, 1024, 1024) -> q: (batch, n_actions)"""
        env_flat = self.env_encoder(x).view(x.size(0), -1)

        if self._clutter_features is not None:
            clutter_flat = self._clutter_features.expand(
                x.size(0), -1, -1, -1).reshape(x.size(0), -1)
            combined = torch.cat([env_flat, clutter_flat], dim=1)
        else:
            combined = env_flat

        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ══════════════════════════════════════════════════════════════════════
# N-Step Return Buffer
# ══════════════════════════════════════════════════════════════════════
# Accumulates N transitions, computes discounted return, then stores
# a single (s_0, a_0, R_n, s_n, terminal) tuple.

class NStepBuffer:
    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def append(self, transition):
        """Add transition. Returns completed N-step tuple or None."""
        self.buffer.append(transition)
        if len(self.buffer) < self.n_step:
            return None

        n_step_return = sum(
            (self.gamma ** i) * (r.item() if hasattr(r, 'item') else r)
            for i, (_, _, r, _, _) in enumerate(self.buffer))

        first, last = self.buffer[0], self.buffer[-1]
        return (first[0], first[1], torch.tensor([[n_step_return]]), last[3], last[4])

    def flush(self):
        """Flush partial transitions at episode end."""
        results = []
        while len(self.buffer) > 0:
            n_step_reward = sum(
                (self.gamma ** i) * (r.item() if hasattr(r, 'item') else r)
                for i, (_, _, r, _, _) in enumerate(self.buffer))
            first, last = self.buffer[0], self.buffer[-1]
            results.append((first[0], first[1], torch.tensor([[n_step_reward]]), last[3], last[4]))
            self.buffer.popleft()
        return results


# ══════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════
# Double DQN + N-step returns + mixed precision.
# CuPy grids stay fp64; states cast to fp32 at the boundary.

def train_qlearner(antennas, antennas_old,
                   population_environment,
                   model_path="qlearner.cpt",
                   antenna_path="antennas.pkl",
                   clutter_dir=None):
    """
    Train the Q-learner.

    Args:
        antennas:              list of Antenna objects
        antennas_old:          copy of antennas (for rollback tracking)
        population_environment: population density grid (1024x1024)
        clutter_dir:           path to directory with clutter_4ch.npy (optional)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build footprint cache (batch-computes all antennas, builds layers)
    cache = FootprintCache(antennas)

    use_clutter = clutter_dir is not None and os.path.isfile(
        os.path.join(clutter_dir, 'clutter_4ch.npy'))

    model = PropogatorCNN2(max_antennas=len(antennas), use_clutter=use_clutter).float().to(device)

    if use_clutter:
        clutter_np = npx.load(os.path.join(clutter_dir, 'clutter_4ch.npy'))
        clutter_tensor = torch.from_numpy(clutter_np).unsqueeze(0).float().to(device)
        model.precompute_clutter(clutter_tensor)
        del clutter_tensor

    target_model = copy.deepcopy(model)
    target_model.eval()

    optimizer = Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    if os.path.isfile(model_path):
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['model_state_dict'])
        target_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

    criterion = nn.SmoothL1Loss()
    experience = deque(maxlen=model.experience_size)
    n_step_buffer = NStepBuffer(n_step=N_STEP, gamma=model.gamma)

    epsilon_decrements = np.linspace(
        model.start_epsilon, model.end_epsilon,
        num=model.n_iters, endpoint=True, retstep=False, dtype=None, axis=0)

    cpu = torch.device('cpu')

    # State init: fp64 -> fp32 at boundary
    image_2, _, _ = frame_step(0, antennas, antennas_old, cache, population_environment)
    image_1 = torch.from_numpy(np.asnumpy(image_2)).float()
    state = torch.cat([image_1] * 8).view(1, 8, max_w, max_l).to(device)

    iteration = 0
    epsilon = epsilon_decrements[0]
    MIN_REPLAY_SIZE = 256
    gamma_n = model.gamma ** N_STEP

    while iteration < model.n_iters:

        with autocast():
            output = model(state)[0]

        action = torch.zeros([model.n_actions])
        random_action = np.random.random() <= epsilon
        action_index = (torch.randint(model.n_actions, torch.Size([]), dtype=torch.int)
                        if random_action else torch.argmax(output))
        action[action_index] = 1

        image_1, reward, terminal = frame_step(
            action_index.item(), antennas, antennas_old, cache, population_environment)

        if terminal == 1:
            continue

        image_data_1 = torch.from_numpy(np.asnumpy(image_1)).float().unsqueeze(0).to(device)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward_tensor = torch.from_numpy(
            npx.array([reward.get() if hasattr(reward, 'get') else reward])).unsqueeze(0)

        raw_transition = (state.to(cpu), action, reward_tensor, state_1.to(cpu), terminal)
        n_step_transition = n_step_buffer.append(raw_transition)

        if n_step_transition is not None:
            experience.append(n_step_transition)

        if len(experience) < MIN_REPLAY_SIZE or iteration % 64 != 0:
            state = state_1
            iteration += 1
            epsilon = epsilon_decrements[min(iteration, len(epsilon_decrements) - 1)]
            continue

        print(f"Iteration {iteration}, epsilon: {epsilon:.4f}, replay: {len(experience)}")

        total_loss = 0
        for _ in range(64):
            epsilon = epsilon_decrements[min(iteration, len(epsilon_decrements) - 1)]
            minibatch = random.sample(list(experience), model.mini_batch_size)

            state_batch = torch.cat(tuple(d[0] for d in minibatch)).to(device)
            action_batch = torch.cat(tuple(d[1] for d in minibatch)).to(device)
            reward_batch = torch.cat(tuple(d[2] for d in minibatch)).to(device)
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch)).to(device)

            # Double Q-Learning targets with N-step bootstrap
            with torch.no_grad():
                with autocast():
                    best_actions = model(state_1_batch).argmax(dim=1, keepdim=True)
                    target_q_values = target_model(state_1_batch).gather(1, best_actions).squeeze(1)

                y_batch = torch.cat(tuple(
                    reward_batch[i] if minibatch[i][4]
                    else reward_batch[i] + gamma_n * target_q_values[i].float().unsqueeze(0)
                    for i in range(len(minibatch))))

            # Mixed precision forward + backward
            optimizer.zero_grad()
            with autocast():
                q_value = torch.sum(model(state_batch) * action_batch, dim=1)
                loss = criterion(q_value, y_batch)

            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Soft-update target network
            with torch.no_grad():
                for tp, mp in zip(target_model.parameters(), model.parameters()):
                    tp.data.copy_(model.tau * mp.data + (1 - model.tau) * tp.data)

            del state_batch, action_batch, reward_batch, state_1_batch, y_batch, q_value, loss
            torch.cuda.empty_cache()

        print(f"  Avg loss: {total_loss / 64:.6f}")

        savepickle("environment_state.pkl", cache.serving_layer)
        savepickle("gantennas.pkl", antennas)

        torch.save({
            'model_state_dict': model.state_dict(),
            'target_model_state_dict': target_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'iteration': iteration,
        }, model_path)

        state = state_1
        iteration += 1


# ══════════════════════════════════════════════════════════════════════
# Simulated Annealing Local Optimizer
# ══════════════════════════════════════════════════════════════════════
# Accepts worse solutions with probability exp(-delta/T) to escape local
# optima. Adaptive step size shrinks with temperature.
# Global best tracked separately (SA can move away from it).

def supervisor_function_sa(objective, bounds, n_iterations, step_size, antennas,
                           orig_solution, pre_calc=None, pre_calc2=None,
                           T_start=1.0, T_end=0.001, tilt_bound=4.0):
    """Simulated annealing for antenna tilt/azimuth optimization."""
    solution = np.zeros(len(antennas))
    for i in range(len(antennas)):
        solution[i] = antennas[i].e_tilt

    solution_eval = objective(solution, antennas, pre_calc, pre_calc2)
    best_solution = solution.copy()
    best_eval = solution_eval
    history = {}

    temperatures = np.logspace(np.log10(T_start), np.log10(T_end), num=n_iterations)

    for i in range(n_iterations):
        T = temperatures[i]
        adaptive_step = step_size * (T / T_start)

        candidate = solution + (np.random.rand(len(bounds)) - 0.5) * adaptive_step
        candidate[candidate > 0.0] = 0.0
        candidate[candidate < -10.0] = -10.0

        candidate[orig_solution - candidate < -tilt_bound] = \
            orig_solution[orig_solution - candidate < -tilt_bound] + tilt_bound
        candidate[orig_solution - candidate > tilt_bound] = \
            orig_solution[orig_solution - candidate > tilt_bound] - tilt_bound

        candidate_eval = objective(candidate, antennas)
        delta = candidate_eval - solution_eval

        if delta <= 0 or np.random.random() < np.exp(-delta / T):
            solution, solution_eval = candidate, candidate_eval
            if solution_eval <= best_eval:
                best_solution, best_eval = solution.copy(), solution_eval
                savepickle("recommendations.pkl", np.asnumpy(solution))
                print(f'>{i} T={T:.4f} f(score) = {solution_eval:.5f} [BEST]')

            history[i] = solution_eval.get() if hasattr(solution_eval, 'get') else solution_eval

    return [best_solution, best_eval, history]


# ══════════════════════════════════════════════════════════════════════
# PPM Pipeline (Small-Sample Aware)
# ══════════════════════════════════════════════════════════════════════
# Validates tilt changes against real UE measurements (2-week windows).
# t-distribution posteriors, paired hourly comparison, adaptive thresholds.
# Monitors target cell + first-tier neighbors. Auto-rollback.

class PPMSinrRsrqTracker:
    """Post-Performance Monitor for SINR and RSRQ KPIs."""

    def __init__(self, go_back_hours=7*24, check_ahead_hours=7*24, sigma=3,
                 busy_hours=(7, 21), weekday_only=False):
        self.go_back_hours = go_back_hours
        self.check_ahead_hours = check_ahead_hours
        self.sigma = sigma
        self.busy_hours = busy_hours
        self.weekday_only = weekday_only
        self.sinr_rsrq_kpis = [
            'Serving Cell Average RSRP', 'Serving Cell Average RSRQ',
            'SINR_p5', 'SINR_p50', 'SINR_p95', 'RSRQ_p5', 'RSRQ_p50',
        ]

    def _filter_hours(self, df):
        filtered = df.copy()
        if self.busy_hours is not None:
            h = filtered['Hour of period_start_time2'].dt.hour
            filtered = filtered[(h >= self.busy_hours[0]) & (h < self.busy_hours[1])]
        if self.weekday_only:
            filtered = filtered[filtered['Hour of period_start_time2'].dt.dayofweek < 5]
        return filtered

    def _pair_by_hour_of_week(self, pre_data, post_data, pre_ts, post_ts):
        """Pair by hour-of-week. Returns (differences, n_pairs)."""
        def _build_lookup(data, ts):
            how = ts.dt.dayofweek * 24 + ts.dt.hour
            lookup = {}
            for h, v in zip(how, data):
                lookup.setdefault(int(h), []).append(v)
            return {k: npx.median(v) for k, v in lookup.items()}

        pre_lk, post_lk = _build_lookup(pre_data, pre_ts), _build_lookup(post_data, post_ts)
        diffs = [post_lk[h] - pre_lk[h] for h in sorted(set(pre_lk) & set(post_lk))]
        return npx.array(diffs), len(diffs)

    def compute_bayesian_comparison(self, pre_data, post_data,
                                    pre_timestamps=None, post_timestamps=None,
                                    n_samples=10000):
        """t-distribution Bayesian before/after test. Tries paired first."""
        from scipy import stats as sp_stats

        pre, post = npx.array(pre_data, dtype=float), npx.array(post_data, dtype=float)

        if pre_timestamps is not None and post_timestamps is not None:
            diffs, n_pairs = self._pair_by_hour_of_week(pre, post, pre_timestamps, post_timestamps)
            if n_pairs >= 10:
                se = diffs.std(ddof=1) / npx.sqrt(n_pairs)
                lift = sp_stats.t(df=n_pairs-1, loc=diffs.mean(), scale=se).rvs(n_samples)
                return {
                    'p_improvement': float((lift > 0).mean()),
                    'expected_lift': float(lift.mean()),
                    'ci_lower': float(npx.percentile(lift, 5)),
                    'ci_upper': float(npx.percentile(lift, 95)),
                    'pre_mean': float(pre.mean()), 'post_mean': float(post.mean()),
                    'n_effective': n_pairs, 'method': 'paired',
                }

        pre_m, pre_s, pre_n = pre.mean(), max(pre.std(ddof=1), 1e-10), len(pre)
        post_m, post_s, post_n = post.mean(), max(post.std(ddof=1), 1e-10), len(post)
        lift = (sp_stats.t(df=max(post_n-1,1), loc=post_m, scale=post_s/npx.sqrt(post_n)).rvs(n_samples)
              - sp_stats.t(df=max(pre_n-1,1), loc=pre_m, scale=pre_s/npx.sqrt(pre_n)).rvs(n_samples))

        return {
            'p_improvement': float((lift > 0).mean()),
            'expected_lift': float(lift.mean()),
            'ci_lower': float(npx.percentile(lift, 5)),
            'ci_upper': float(npx.percentile(lift, 95)),
            'pre_mean': float(pre_m), 'post_mean': float(post_m),
            'n_effective': min(pre_n, post_n), 'method': 'unpaired',
        }

    def evaluate_tilt_change(self, df, cell_name, change_timestamp, neighbor_cells=None):
        """Evaluate tilt change on target cell and neighbors."""
        results = {'target_cell': {}, 'neighbor_impact': {}}

        cell_data = self._filter_hours(
            df[df['Cell Name'] == cell_name].sort_values('Hour of period_start_time2'))

        for kpi in self.sinr_rsrq_kpis:
            if kpi not in cell_data.columns:
                continue
            pre = cell_data[cell_data['Hour of period_start_time2'] < change_timestamp].dropna(subset=[kpi])
            post = cell_data[cell_data['Hour of period_start_time2'] >= change_timestamp].dropna(subset=[kpi])
            if len(pre) < 10 or len(post) < 10:
                continue

            bayes = self.compute_bayesian_comparison(
                pre[kpi].values, post[kpi].values,
                pre['Hour of period_start_time2'], post['Hour of period_start_time2'])

            pre_vals = pre[kpi].values
            pm, ps = pre_vals.mean(), pre_vals.std()
            post_vals = post[kpi].values

            results['target_cell'][kpi] = {
                'bayesian': bayes,
                'sigma_uplift': float((post_vals > pm + self.sigma * ps).sum() / len(post_vals)),
                'sigma_downshift': float((post_vals < pm - self.sigma * ps).sum() / len(post_vals)),
                'n_pre': len(pre), 'n_post': len(post),
            }

        if neighbor_cells:
            for nb in neighbor_cells:
                nb_data = self._filter_hours(
                    df[df['Cell Name'] == nb].sort_values('Hour of period_start_time2'))
                results['neighbor_impact'][nb] = {}
                for kpi in self.sinr_rsrq_kpis:
                    if kpi not in nb_data.columns:
                        continue
                    pre = nb_data[nb_data['Hour of period_start_time2'] < change_timestamp].dropna(subset=[kpi])
                    post = nb_data[nb_data['Hour of period_start_time2'] >= change_timestamp].dropna(subset=[kpi])
                    if len(pre) < 10 or len(post) < 10:
                        continue
                    results['neighbor_impact'][nb][kpi] = self.compute_bayesian_comparison(
                        pre[kpi].values, post[kpi].values,
                        pre['Hour of period_start_time2'], post['Hour of period_start_time2'])

        return results

    def _adaptive_threshold(self, n):
        """Decision threshold calibrated to sample size."""
        if n >= 150: return 0.85
        if n >= 90:  return 0.82
        if n >= 40:  return 0.80
        return 0.75

    def check_rollback(self, results, degradation_threshold=None):
        """Check if tilt change should be rolled back."""
        critical = ['RSRQ_p5', 'SINR_p5', 'Serving Cell Average RSRQ']
        rollback, reasons = False, []

        for kpi in critical:
            if kpi not in results['target_cell']:
                continue
            b = results['target_cell'][kpi]['bayesian']
            thr = degradation_threshold or self._adaptive_threshold(b.get('n_effective', 168))
            p_deg = 1 - b['p_improvement']
            if p_deg > thr:
                rollback = True
                reasons.append(f"Target {kpi}: P(deg)={p_deg:.1%} (thr={thr:.0%}, N={b['n_effective']})")

        for nb, kpis in results['neighbor_impact'].items():
            for kpi in critical:
                if kpi not in kpis:
                    continue
                b = kpis[kpi]
                thr = degradation_threshold or self._adaptive_threshold(b.get('n_effective', 168))
                p_deg = 1 - b['p_improvement']
                if p_deg > thr:
                    rollback = True
                    reasons.append(f"Neighbor {nb} {kpi}: P(deg)={p_deg:.1%} (thr={thr:.0%})")

        return {'rollback_recommended': rollback, 'reasons': reasons}


# ══════════════════════════════════════════════════════════════════════
# Hardware Specifications
# ══════════════════════════════════════════════════════════════════════
#
# GPU:    T4 (16 GB) min, V100 recommended, A100 for all-Australia.
#         CuPy requires CUDA. Peak ~1.5-2 GB at minibatch 8.
# CPU:    4+ cores. GCP n1-standard-4 minimum.
# RAM:    16 GB min, 32 GB recommended. Replay buffer ~4-6 GB.
# Disk:   ~10 GB working space.
#
# Per-step timing (T4, 200 antennas):
#   make_norm (local patch):  ~1.2 ms   (was ~5 ms full grid)
#   cache.update_antenna:     ~1.8 ms   (incremental layer rebuild)
#   CNN forward (fp16):       ~2 ms     (autocast tensor cores)
#   Total per step:           ~5 ms     (was ~17 ms)
#
# Init timing (200 antennas):
#   FootprintCache build:     ~50 ms    (batched, was ~1s sequential)
#
# Training (100K iters, minibatch 8):
#   T4: ~1.5-3 hr    V100: ~0.5-1.5 hr    A100: ~0.3-0.7 hr
#
# Precision: fp32 weights, fp16 forward (autocast), fp64 CuPy grids.
#
# GCP instances:
#   Dev:  n1-standard-4 + T4    ~$1.40/hr
#   Prod: n1-highmem-8 + V100   ~$3.50/hr
#   Full: n1-highmem-8 + A100   ~$5.00/hr
