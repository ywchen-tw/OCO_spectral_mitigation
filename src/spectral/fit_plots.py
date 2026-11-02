"""Example-fit visualisations for the spectral pipeline.

plot_doppler_fit_example: one PNG per band per orbit for the Doppler-variant
fit set (selected in fitting.process_orbit since 2026-10-31).
plot_fitting_example / plot_orbit_fitting_examples: the older production-order
(Savitzky-Golay) examples, kept for fitting_with_ring_effect and old callers.

Split out of fitting.py (2026-07, review §7.4); fitting.py re-exports the
public names.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from .cumulant_fit import (LOG_TRANSMITTANCE_MODELS, compute_transmittance,
                           fit_spectral_model)


def plot_fitting_example(tag, fp, sounding_ind, wvl, rad, transmittance, tau, ln_T, popt_log, fit_order, output_dir):
    """Save example scatter plots of the cumulant and gamma-model fits.

    Produces two PNG files per call:
      - {tag}_log_T_fit_fp{fp}_snd{snd}.png   : ln(T) vs tau with polynomial fit
      - {tag}_T_fit_fp{fp}_snd{snd}.png        : T vs tau with gamma-dist fit
    """
    model_func  = LOG_TRANSMITTANCE_MODELS[fit_order]
    sort_idx    = np.argsort(tau)
    tau_sorted  = tau[sort_idx]
    ln_T_smooth = savgol_filter(ln_T[sort_idx], window_length=31, polyorder=3)
    tau_fit     = np.linspace(tau.min(), tau.max(), 100)

    kappa_1 = popt_log[0]
    kappa_2 = popt_log[1] if fit_order >= 2 else 0.0

    _FONT        = "Arial"
    _LABEL_FS    = 20
    _TICK_FS     = 17
    _LEGEND_FS   = 17
    _TITLE_FS    = 20
    _SUPTITLE_FS = 24

    _rc = {
        "font.family":     _FONT,
        "axes.labelsize":  _LABEL_FS,
        "axes.titlesize":  _TITLE_FS,
        "legend.fontsize": _LEGEND_FS,
        "xtick.labelsize": _TICK_FS,
        "ytick.labelsize": _TICK_FS,
    }
    with plt.rc_context(_rc):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        ax1r = ax1.twinx()
        l1 = ax1.plot(wvl, rad, label="Radiance", color="green", linewidth=2.5)
        l2 = ax1r.plot(wvl, transmittance, label="Transmittance", color="blue")
        ax1.set(xlabel="Wavelength (nm)", ylabel="Radiance")
        ax1r.set(ylabel="Transmittance")
        ax1.legend(l1 + l2, [l.get_label() for l in l1 + l2],
                   loc="lower left", bbox_to_anchor=(0.18, 1.02),
                   ncol=2, borderaxespad=0)
        ax2.scatter(tau, ln_T, label="Observed", color="blue", s=10)
        ax2.plot(tau_fit, model_func(tau_fit, *popt_log), label="Fitted", color="red")
        ax2.plot(tau_sorted, ln_T_smooth, label="Smoothed Observed", color="orange", alpha=0.7)
        ax2.set(
            xlabel=f"Total {tag.upper()} Optical Depth",
            ylabel="ln(Transmittance)",
        )
        ax2.set_title(f"κ1: {kappa_1:.3e}  κ2: {kappa_2:.3e}", pad=14)
        ax2.legend()
        fig.suptitle(f"FP {fp}  Sounding {sounding_ind}",
                     fontsize=_SUPTITLE_FS, y=1.05)
        fig.savefig(
            f"{output_dir}/{tag}_log_T_fit_fp{fp}_snd{sounding_ind}.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # # Plot 2: T vs tau using the gamma-distribution model
    # try:
    #     popt2, _ = curve_fit(transmittance_model, tau, np.exp(ln_T))
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.scatter(tau, np.exp(ln_T), label="Observed", color="blue", s=10)
    #     ax.plot(tau_fit, transmittance_model(tau_fit, *popt2), label="Fitted (gamma)", color="red")
    #     ax.set(
    #         xlabel=f"Total {tag.upper()} Optical Depth",
    #         ylabel="Transmittance",
    #         title=(f"FP {fp}  Sounding {sounding_ind}\n"
    #                f"κ₁: {popt2[0]:.3e}  κ₂: {popt2[1]:.3e}  intercept: {popt2[2]:.3e}"),
    #     )
    #     ax.legend()
    #     fig.savefig(
    #         f"{output_dir}/{tag}_T_fit_fp{fp}_snd{sounding_ind}.png",
    #         dpi=150, bbox_inches="tight",
    #     )
    #     plt.close(fig)
    # except (RuntimeError, ValueError):
    #     pass  # Gamma model may not converge for every sounding; skip quietly


def _doppler_predictions(i_band, tau, td, tw, x, p):
    """ln T predicted by the stored order-2, order-3 and agreed-model fits.

    p : {field: value} for one sounding and band (cumulant_fit.FIT_FIELDS).
    """
    pred = {
        "order 2": p["c_o2"] - p["k1_o2"] * tau + 0.5 * p["k2_o2"] * tau**2,
        "order 3": (p["c_o3"] - p["k1_o3"] * tau + 0.5 * p["k2_o3"] * tau**2
                    - p["k3_o3"] / 3.0 * tau**3),
    }
    base = p["c_am"] + p["a1_am"] * x
    if i_band == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            pred["agreed model"] = base + np.log((1 - p["f_am"]) * np.exp(-p["k1_am"] * tau) + p["f_am"])
    else:
        am = base - p["k1_am"] * td - p["k1w_am"] * tw + 0.5 * p["k2_am"] * tau**2
        if i_band == 2 and np.isfinite(p["a2_am"]):
            am = am + p["a2_am"] * x**2
        pred["agreed model"] = am
    return pred


def plot_doppler_fit_example(tag, i_band, fp, sounding_id, wvl, rad, transmittance,
                             tau, ln_T, frac, x, p, output_dir):
    """One example PNG for a band: spectrum (left) and ln T vs τ with the
    stored order-2, order-3 and agreed-model fits evaluated at the observed
    channels (right; raw ln T, no smoothing).  The agreed model depends on the
    channel position and τ split, so it is drawn as points, not a curve.

    tau, ln_T, frac, x : the fitted channels (finite, edge channels excluded)
    p                  : {field: value} for this sounding and band (FIT_FIELDS)
    """
    tw = frac * tau
    td = tau - tw
    pred = _doppler_predictions(i_band, tau, td, tw, x, p)
    order = np.argsort(tau)
    rc = {"font.family": ["Arial", "DejaVu Sans"], "axes.labelsize": 16,
          "axes.titlesize": 15, "legend.fontsize": 13,
          "xtick.labelsize": 13, "ytick.labelsize": 13}
    with plt.rc_context(rc):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        ax1r = ax1.twinx()
        l1 = ax1.plot(wvl, rad, label="Radiance", color="green", linewidth=2)
        l2 = ax1r.plot(wvl, transmittance, label="Transmittance", color="blue", linewidth=1)
        ax1.set(xlabel="Wavelength (nm)", ylabel="Radiance")
        ax1r.set(ylabel="Transmittance")
        ax1.legend(l1 + l2, [line.get_label() for line in l1 + l2], loc="lower left")
        ax2.scatter(tau, ln_T, label="Observed", color="0.55", s=8)
        ax2.plot(tau[order], pred["order 2"][order], color="tab:orange", label="Order 2")
        ax2.plot(tau[order], pred["order 3"][order], color="tab:purple", ls="--", label="Order 3")
        ax2.scatter(tau, pred["agreed model"], color="tab:red", s=4, label="Agreed model")
        ax2.set(xlabel=f"Total {tag.upper()} optical depth", ylabel="ln(Transmittance)")
        ax2.set_title(f"k1_am = {p['k1_am']:.4f} ± {p['sig_k1_am']:.1e}   "
                      f"rms_am = {p['rms_am']:.2e}", pad=10)
        ax2.legend()
        fig.suptitle(f"{tag.upper()}  FP {fp}  sounding {sounding_id}", fontsize=17)
        fig.savefig(f"{output_dir}/{tag}_fit_example_snd{sounding_id}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_orbit_fitting_examples(od, fit_orders, output_dir):
    """Save one representative fitting example per band for an orbit."""
    os.makedirs(output_dir, exist_ok=True)
    tags = ["o2a", "wco2", "sco2"]
    T_all = compute_transmittance(od["radiances"], od["toa_sol"])
    ln_T_all = np.where(T_all > 0, np.log(T_all), np.nan)
    plot_done = {tag: False for tag in tags}

    for j in range(len(od["sounding_id"])):
        if all(plot_done.values()):
            return
        if not od["valid_l1b"][j]:
            continue

        for i_band, (tag, band_order) in enumerate(zip(tags, fit_orders)):
            if plot_done[tag]:
                continue

            tau_j = od["tau"][i_band, j][1:-1]
            ln_T_j = ln_T_all[i_band, j][1:-1]
            mask = ~np.isnan(ln_T_j) & ~np.isnan(tau_j)
            if mask.sum() < band_order + 2:
                continue

            try:
                popt = fit_spectral_model(tau_j[mask], ln_T_j[mask], band_order)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue

            plot_fitting_example(
                tag, int(od["fp_number"][j]), int(od["sounding_id"][j]),
                od["wvl"][i_band, od["fp_number"][j]],
                od["radiances"][i_band, j], T_all[i_band, j],
                tau_j[mask], ln_T_j[mask], popt, band_order, output_dir,
            )
            plot_done[tag] = True

