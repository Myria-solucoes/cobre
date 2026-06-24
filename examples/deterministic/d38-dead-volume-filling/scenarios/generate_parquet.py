"""
Generate Parquet scenario files for the d38-dead-volume-filling test case.

Run with: uv tool run --with pyarrow -- python3 generate_parquet.py

Produces:
  - inflow_seasonal_stats.parquet
  - load_seasonal_stats.parquet

Determinism: std_m3s = 0.0 and std_mw = 0.0 on every row, so each realized
inflow/load equals its mean exactly (no per-opening noise) and the simulated
trajectory is bit-reproducible.

Topology / binding regime
=========================
Cascade H1 (id 0) -> H2 (id 1, filling) -> H3 (id 2, real fed downstream),
plus off-cascade control H4 (id 3). H2 fills from start_stage_id=2 to
entry_stage_id=4 (PreFilling 0-1, Filling 2-3, Operating 4-5).

H2's mean_m3s schedule is the crux:
  - Filling (stages 2-3): SHORT (5.0 m3/s) so even at the per-stage impound
    cap (filling_min_rate_m3s = 10.0, cap rise = zeta * 10 = 25.92 hm3/stage with
    zeta = 720 h * 0.0036) the reservoir cannot reach the dead volume
    (min_storage_hm3 = 60.0) by the terminal Filling stage (id 3): the cap alone
    bounds two-stage accumulation to 51.84 hm3 < 60, and the short inflow holds it
    even lower, so the terminal target slack sigma_fill > 0 at stage 3.
  - Operating (stages 4-5): RECOVERS (60.0 m3/s) so H2 climbs above the dead
    volume and the soft operating floor stops binding: sigma^{v-} -> 0 at stage 5.

H1 carries enough inflow to release water into the cascade during PreFilling so
the short-circuit routes a visible contribution onto H3's real water-balance row.
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))

N_STAGES = 6

# ── inflow_seasonal_stats.parquet ─────────────────────────────────────────────
#
# Schema: hydro_id INT32, stage_id INT32, mean_m3s FLOAT64, std_m3s FLOAT64
#
# Per-(hydro, stage) mean schedule; std is 0.0 everywhere (deterministic).
#
#         stage:    0     1     2     3     4     5
# H1 (0):          30    30    30    30    30    30   (cascade feeder)
# H2 (1):          15    15     5     5    60    60   (short in Filling, recovers)
# H3 (2):          20    20    20    20    20    20   (own incremental)
# H4 (3):          25    25    25    25    25    25   (off-cascade control)
inflow_means = {
    0: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
    1: [15.0, 15.0, 5.0, 5.0, 60.0, 60.0],
    2: [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    3: [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
}

inflow_hydro_ids = []
inflow_stage_ids = []
inflow_mean_m3s = []
inflow_std_m3s = []
for hydro_id in sorted(inflow_means):
    for stage_id in range(N_STAGES):
        inflow_hydro_ids.append(hydro_id)
        inflow_stage_ids.append(stage_id)
        inflow_mean_m3s.append(inflow_means[hydro_id][stage_id])
        inflow_std_m3s.append(0.0)

inflow_schema = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("mean_m3s", pa.float64(), nullable=False),
        pa.field("std_m3s", pa.float64(), nullable=False),
    ]
)

inflow_table = pa.table(
    {
        "hydro_id": pa.array(inflow_hydro_ids, type=pa.int32()),
        "stage_id": pa.array(inflow_stage_ids, type=pa.int32()),
        "mean_m3s": pa.array(inflow_mean_m3s, type=pa.float64()),
        "std_m3s": pa.array(inflow_std_m3s, type=pa.float64()),
    },
    schema=inflow_schema,
)

inflow_path = os.path.join(script_dir, "inflow_seasonal_stats.parquet")
pq.write_table(inflow_table, inflow_path, compression="zstd")
print(f"wrote {len(inflow_table)} rows -> {inflow_path}")

# ── load_seasonal_stats.parquet ───────────────────────────────────────────────
#
# Schema: bus_id INT32, stage_id INT32, mean_mw FLOAT64, std_mw FLOAT64
#
# One bus (id 0); constant 150 MW mean; std 0.0 (deterministic). Per-block shape
# comes from load_factors.json (sized to each stage's block count).
load_bus_ids = [0] * N_STAGES
load_stage_ids = list(range(N_STAGES))
load_mean_mw = [150.0] * N_STAGES
load_std_mw = [0.0] * N_STAGES

load_schema = pa.schema(
    [
        pa.field("bus_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("mean_mw", pa.float64(), nullable=False),
        pa.field("std_mw", pa.float64(), nullable=False),
    ]
)

load_table = pa.table(
    {
        "bus_id": pa.array(load_bus_ids, type=pa.int32()),
        "stage_id": pa.array(load_stage_ids, type=pa.int32()),
        "mean_mw": pa.array(load_mean_mw, type=pa.float64()),
        "std_mw": pa.array(load_std_mw, type=pa.float64()),
    },
    schema=load_schema,
)

load_path = os.path.join(script_dir, "load_seasonal_stats.parquet")
pq.write_table(load_table, load_path, compression="zstd")
print(f"wrote {len(load_table)} rows -> {load_path}")

print("done.")
