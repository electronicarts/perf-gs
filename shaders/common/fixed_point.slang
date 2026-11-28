// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_FIXED_POINT_SLANG
#define TRAINER_FIXED_POINT_SLANG

// Fixed-point atomic gradient accumulation. Backends without 64-bit integer
// buffer atomics (e.g. MoltenVK) compile with FIXED_POINT_INT32 using 32-bit
// atomics; other backends retain the full-range 64-bit accumulator.
// Scale 4096 keeps 1/4096 gradient resolution while allowing signed sums up to
// about 524k before overflow. It completed the 3,000-step M2 Truck run; larger
// or higher-gradient scenes have not been stress-tested for accumulator range.
#ifdef FIXED_POINT_INT32
typealias fixed_t = int;
static const float FIXED_SCALE = 4096.0;
// Removal scores sum squared RGB energy, not linear gradients, so use an
// independent scale with a larger 32-bit headroom. Keep the 65,536x scale ratio
// to the 64-bit path; this is not the gradient FIXED_SCALE.
static const float REMOVAL_COST_SCALE = 16384.0;
#else
typealias fixed_t = int64_t;
static const float FIXED_SCALE = 65536.0;
static const float REMOVAL_COST_SCALE = 1073741824.0;
#endif
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;
// Per-axis structure scores (eta_max) use uint InterlockedMax on all backends, independent of fixed_t width.
static const float ETA_FIXED_SCALE = 65536.0;

fixed_t toFixed(float value) { return fixed_t(value * FIXED_SCALE); }
float fromFixed(fixed_t value) { return float(value) * FIXED_INV_SCALE; }

#endif // TRAINER_FIXED_POINT_SLANG
