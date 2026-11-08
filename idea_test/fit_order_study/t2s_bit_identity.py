"""Before/after bit-identity of the production fit code on synthetic soundings (t2s test, 2026-11-02).

Compares spectral/cumulant_fit.py against a baseline copy of the module:
constants (FIT_SCHEMA, FIT_FIELDS, EXTRA_FIT_FIELDS, ...) must be equal, every
baseline function's source is compared (a difference is reported; with the
HEAD baseline the 2026-11-02 rename _fit_am_o2a -> fit_am_o2a shows up in
three functions), and the outputs of _doppler_fit_chunk, _extra_fit_chunk
(with and without frac), _fit_chunk and the _am entry points on synthetic
float64 and float32 chunks must be bitwise equal (NaN == NaN).  Also checks
that t2s reproduces SCO2 am and WCO2 k2diag bitwise.

Usage:
    git show HEAD:src/spectral/cumulant_fit.py > /tmp/cf_head.py
    python idea_test/fit_order_study/t2s_bit_identity.py /tmp/cf_head.py
"""
import hashlib, importlib.util, inspect, sys
import numpy as np

REPO = "/Volumes/YWC_SSD/programming/oco_fp_analysis"
BEFORE = sys.argv[1]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

old = load(BEFORE, "cf_before")
new = load(f"{REPO}/src/spectral/cumulant_fit.py", "cf_after")

# 1. constants
for name in ("FIT_SCHEMA", "FIT_FIELDS", "EXTRA_FIT_FIELDS", "_AM_FIELDS", "_T2_FIELDS",
             "_AM_MIN_CHAN", "_AM_F_MAX", "_X_HALF", "_SLAB_CHANNEL_OFFSET", "_FIT_INDEX"):
    assert getattr(old, name) == getattr(new, name), name
print("constants identical")

# 2. source text of every pre-existing function
names = [n for n, o in vars(old).items() if inspect.isfunction(o) and o.__module__ == "cf_before"]
for n in names:
    a, b = inspect.getsource(getattr(old, n)), inspect.getsource(getattr(new, n))
    if a != b:
        print("  source differs from baseline:", n)
print(f"source compared for {len(names)} baseline functions")

# 3. numerical outputs on synthetic production-shaped chunks
rng = np.random.default_rng(20261102)
m, nch = 120, 1014
j = np.arange(m)
tau = np.empty((3, m, nch)); frac = np.empty((3, m, nch)); lnT = np.empty((3, m, nch))
xx = (np.arange(nch) + 1 - 507.5) / 507.5
for b in range(3):
    tau[b] = rng.gamma(0.6, 1.0, (m, nch)) * (1 + b)
    frac[b] = rng.uniform(0, 0.3 if b else 0.02, (m, nch))
    k1 = rng.uniform(0.6, 1.2, (m, 1)); k2 = rng.uniform(-0.05, 0.1, (m, 1)); f = rng.uniform(0, 0.08, (m, 1))
    T = (1 - f) * np.exp(-k1 * tau[b] + 0.5 * k2 * np.minimum(tau[b], 3) ** 2) + f
    T *= np.exp(rng.uniform(-1.5, -0.5, (m, 1)) + 0.01 * xx) * (1 + 0.003 * rng.standard_normal((m, nch)))
    T[T > 1] = np.nan
    lnT[b] = np.log(T)
# masks, NaN frac, a few starved soundings
lnT[:, ::17, ::3] = np.nan
frac[:, ::5, ::7] = np.nan
lnT[:, 3, 6:] = np.nan          # 6 channels: < 8 -> no t2/am
lnT[:, 4, 3:] = np.nan          # 3 channels: < 5 -> nothing
tau32, ln32, fr32 = (a.astype(np.float32) for a in (tau, lnT, frac))
land = rng.random(m) < 0.5

def same(a, b):
    return np.array_equal(a, b, equal_nan=True)

for label, args in (("float64", (tau, lnT, frac)), ("float32", (tau32, ln32, fr32))):
    ra = old._doppler_fit_chunk(j, *args, land)[1]
    rb = new._doppler_fit_chunk(j, *args, land)[1]
    assert same(ra, rb), label
    ea = old._extra_fit_chunk(j, *args[:2], (7, 3, 7), args[2], land)[1]
    eb = new._extra_fit_chunk(j, *args[:2], (7, 3, 7), args[2], land)[1]
    assert same(ea, eb), label
    ea = old._extra_fit_chunk(j, *args[:2], (7, 3, 7))[1]
    eb = new._extra_fit_chunk(j, *args[:2], (7, 3, 7))[1]
    assert same(ea, eb), label
    fa = old._fit_chunk(j, *args[:2], (7, 3, 7), True)
    fb = new._fit_chunk(j, *args[:2], (7, 3, 7), True)
    assert all(same(a, b) for a, b in zip(fa[:5], fb[:5])), label
    h = hashlib.sha256(np.ascontiguousarray(rb).tobytes()).hexdigest()[:16]
    print(f"{label}: _doppler_fit_chunk {rb.shape} finite {np.isfinite(rb).sum()} "
          f"sha256 {h} identical; _extra_fit_chunk (with/without frac), _fit_chunk identical")

# direct _am entry points
t = tau[0, 0][np.isfinite(lnT[0, 0])]; y = lnT[0, 0][np.isfinite(lnT[0, 0])]; x = xx[np.isfinite(lnT[0, 0])]
for fn, a in (("_fit_am_o2a", (t, x, y)), ("_fit_am_wco2", (0.8 * t, 0.2 * t, x, y)),
              ("_fit_am_sco2", (0.8 * t, 0.2 * t, x, y, True)), ("_fit_am_sco2", (0.8 * t, 0.2 * t, x, y, False))):
    da, db = getattr(old, fn)(*a), getattr(new, fn)(*a)
    assert da.keys() == db.keys() and all(same(np.float64(da[k]), np.float64(db[k])) for k in da), fn
print("fit_am_o2a / _fit_am_wco2 / _fit_am_sco2 (land, water) identical")

# 4. t2s sanity: SCO2 t2s == SCO2 am bitwise; WCO2 k2 == k2diag_am
d = new._doppler_fit_chunk(j, tau32, ln32, fr32, land)[1]
s = new.t2s_fit_chunk(j, tau32, ln32, fr32, land)[1]
F, T = new.FIT_FIELDS.index, new.T2S_FIELDS.index
for stub in ("k1", "k1w", "k2", "a1", "a2", "c", "rms", "sig_k1", "sig_k1w", "sig_k2", "sig_a1", "sig_a2", "sig_c"):
    assert same(d[2, :, F(f"{stub}_am")], s[2, :, T(f"{stub}_t2s")]), stub
assert same(d[1, :, F("k2diag_am")], s[1, :, T("k2_t2s")])
assert same(d[1, :, F("sig_k2diag_am")], s[1, :, T("sig_k2_t2s")])
assert same(d[:, :, F("nchan")], s[:, :, T("nchan")])
print("t2s: SCO2 == SCO2 am (13 fields), WCO2 k2_t2s == k2diag_am, nchan == nchan: bitwise")
print("ALL CHECKS PASSED")
