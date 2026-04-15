"""
simulate_bcp_incidents.py
─────────────────────────────────────────────────────────────────────────────
Business Continuity Recovery Time Analysis — synthetic incident data generator

Produces a DataFrame (and optional CSV) of N simulated incidents stratified
across three incident types: cyber, infrastructure, third-party.

Each row contains:
  - incident_id          : unique identifier
  - incident_type        : cyber | infrastructure | third_party
  - start_dt             : incident start timestamp (pandas Timestamp)
  - outage_hours         : duration of the outage / impact window (hours)
  - recovery_hours       : time from incident start to full recovery (hours)
  - rto_target_hours     : stated RTO target for that incident type (hours)
  - rto_met              : bool — was recovery_hours <= rto_target_hours?
  - censored             : bool — True if incident had not resolved by the
                           observation window end date (essential for KM)
  - severity             : P1 | P2 | P3 — derived from outage_hours thresholds

Distribution choices and parameter calibration
─────────────────────────────────────────────────────────────────────────────
Cyber       Lognormal(μ=2.5, σ=1.2) for outage; Lognormal(μ=3.0, σ=1.5)
            for recovery.  High σ captures the fat tail of ransomware and
            nation-state events (Uptime Institute 2023; ENISA Threat
            Landscape 2023 median ~18h, 95th pct ~240h).

Infra       Weibull(k=1.4, λ=6.0) for outage; Weibull(k=1.6, λ=9.0) for
            recovery.  k > 1 encodes wear-out failure mode; parameters
            calibrated to cloud provider MTTR data (AWS/Azure status
            histories 2020-2024 median ~4h).

Third-party Gamma(α=2.5, β=5.0) + 2h floor for both outage and recovery.
            The constant offset models the irreducible vendor notification
            lag before any recovery action can begin.

Correlation between outage_hours and recovery_hours
─────────────────────────────────────────────────────────────────────────────
Recovery is NOT simply a deterministic function of outage.  We use a
Gaussian copula with ρ = 0.65 to induce positive correlation while
preserving marginal distribution shapes — longer outages tend to have
longer recoveries, but not always (hot-standby / pre-positioned response).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


# ── reproducibility ──────────────────────────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)


# ── simulation parameters ─────────────────────────────────────────────────────
N_INCIDENTS       = 800          # total incidents across all types
OBS_START         = pd.Timestamp("2020-01-01")
OBS_END           = pd.Timestamp("2024-06-30")  # observation window
OBS_HOURS         = (OBS_END - OBS_START).days * 24
SNAPSHOT_DATE = pd.Timestamp("2024-06-30")
ADMIN_CENSOR_RATE = 0.06   # 6% of incidents still open at snapshot

# Incident type mix — loosely consistent with ENISA / Verizon DBIR proportions
TYPE_WEIGHTS      = {"cyber": 0.40, "infrastructure": 0.35, "third_party": 0.25}

# Stated RTO targets (hours) — illustrative fintech firm policy
RTO_TARGETS       = {"cyber": 24.0, "infrastructure": 8.0, "third_party": 24.0}

# Severity thresholds (outage hours) — P1/P2/P3 classification
SEVERITY_THRESHOLDS = {"P1": 8.0, "P2": 2.0}   # >= P1_threshold → P1, etc.

# Copula correlation between outage duration and recovery time
COPULA_RHO        = 0.65


# ── distribution helpers ──────────────────────────────────────────────────────

def correlated_uniform_pair(n: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two correlated Uniform(0,1) samples via a Gaussian copula.
    Maps to marginal distributions using the probability-integral transform.
    """
    cov = [[1.0, rho], [rho, 1.0]]
    z   = rng.multivariate_normal([0, 0], cov, size=n)
    u1  = norm.cdf(z[:, 0])
    u2  = norm.cdf(z[:, 1])
    return u1, u2


def sample_lognormal(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Inverse-CDF sample from Lognormal using pre-computed uniform quantiles."""
    return stats.lognorm.ppf(u, s=sigma, scale=np.exp(mu))


def sample_weibull(u: np.ndarray, k: float, lam: float) -> np.ndarray:
    """Inverse-CDF sample from Weibull(k, λ) using pre-computed uniform quantiles."""
    return stats.weibull_min.ppf(u, c=k, scale=lam)


def sample_gamma_offset(u: np.ndarray, alpha: float, beta: float,
                        offset: float) -> np.ndarray:
    """Inverse-CDF sample from Gamma(α, β) + constant offset."""
    return stats.gamma.ppf(u, a=alpha, scale=beta) + offset


def classify_severity(outage_hours: np.ndarray) -> list[str]:
    """Assign P1/P2/P3 based on outage duration thresholds."""
    labels = []
    for h in outage_hours:
        if h >= SEVERITY_THRESHOLDS["P1"]:
            labels.append("P1")
        elif h >= SEVERITY_THRESHOLDS["P2"]:
            labels.append("P2")
        else:
            labels.append("P3")
    return labels


# ── per-type simulators ───────────────────────────────────────────────────────

def simulate_cyber(n: int) -> pd.DataFrame:
    """
    Cyber incidents: Lognormal tails for both outage and recovery.
    High σ reflects the bimodal reality of cyber events — most resolve
    quickly; a meaningful minority run for days or weeks.
    """
    u_out, u_rec = correlated_uniform_pair(n, COPULA_RHO)
    outage   = sample_lognormal(u_out, mu=2.5, sigma=1.2).clip(min=0.5)
    recovery = sample_lognormal(u_rec, mu=3.0, sigma=1.5).clip(min=outage * 0.8)
    return pd.DataFrame({"outage_hours": outage, "recovery_hours": recovery,
                         "incident_type": "cyber"})


def simulate_infrastructure(n: int) -> pd.DataFrame:
    """
    Infrastructure failures: Weibull with k > 1 (wear-out regime).
    Recovery is also Weibull but with a higher scale — hardware logistics
    add deterministic time that cyber responses don't always have.
    """
    u_out, u_rec = correlated_uniform_pair(n, COPULA_RHO)
    outage   = sample_weibull(u_out, k=1.4, lam=6.0).clip(min=0.25)
    recovery = sample_weibull(u_rec, k=1.6, lam=9.0).clip(min=outage * 0.9)
    return pd.DataFrame({"outage_hours": outage, "recovery_hours": recovery,
                         "incident_type": "infrastructure"})


def simulate_third_party(n: int) -> pd.DataFrame:
    """
    Third-party incidents: Gamma + 2-hour floor.
    The offset models the irreducible vendor notification lag; the Gamma
    shape captures the sum-of-delays structure of third-party escalation.
    """
    u_out, u_rec = correlated_uniform_pair(n, COPULA_RHO)
    outage   = sample_gamma_offset(u_out, alpha=2.5, beta=5.0, offset=2.0).clip(min=0.5)
    recovery = sample_gamma_offset(u_rec, alpha=3.0, beta=6.0, offset=2.0).clip(min=outage * 0.85)
    return pd.DataFrame({"outage_hours": outage, "recovery_hours": recovery,
                         "incident_type": "third_party"})


# ── timestamp generator ───────────────────────────────────────────────────────

def assign_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scatter incident start times uniformly across the observation window.
    Preserves seasonal realism without introducing spurious patterns.
    """
    n              = len(df)
    offset_hours   = rng.uniform(0, OBS_HOURS, size=n)
    df["start_dt"] = OBS_START + pd.to_timedelta(offset_hours, unit="h")
    return df


# ── censoring indicator ───────────────────────────────────────────────────────

def apply_censoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two censoring mechanisms — both are realistic:

    1. Window censoring: incident recovery end-time exceeds SNAPSHOT_DATE.
       Captures incidents that started late in the observation period with
       long recovery tails.

    2. Administrative censoring: a random ~6% of incidents are flagged as
       still open at the snapshot date (analyst pulled data mid-resolution).
       This is the dominant mechanism when recovery times are short relative
       to the observation window.

    For all censored rows, recovery_hours is replaced with the observable
    maximum (snapshot_date - start_dt), not the simulated value.
    """
    recovery_end = df["start_dt"] + pd.to_timedelta(df["recovery_hours"], unit="h")

    # Mechanism 1 — window censoring
    window_censored = recovery_end > SNAPSHOT_DATE

    # Mechanism 2 — administrative censoring (still open at snapshot)
    admin_mask = rng.random(size=len(df)) < ADMIN_CENSOR_RATE
    # Only censor incidents that started before the snapshot
    admin_censored = admin_mask & (df["start_dt"] < SNAPSHOT_DATE)

    censored = window_censored | admin_censored

    # Replace recovery_hours with observable maximum for censored rows
    observable_max = (SNAPSHOT_DATE - df["start_dt"]).dt.total_seconds() / 3600
    observable_max = observable_max.clip(lower=0.5)
    df.loc[censored, "recovery_hours"] = observable_max[censored]
    df["censored"] = censored
    return df


# ── master assembler ──────────────────────────────────────────────────────────

def simulate_incidents(n: int = N_INCIDENTS) -> pd.DataFrame:
    """
    Assemble the full incident dataset.

    Parameters
    ----------
    n : int
        Total number of incidents to simulate.

    Returns
    -------
    pd.DataFrame with columns:
        incident_id, incident_type, start_dt, outage_hours,
        recovery_hours, rto_target_hours, rto_met, censored, severity
    """
    counts = {t: max(1, round(n * w)) for t, w in TYPE_WEIGHTS.items()}

    frames = [
        simulate_cyber(counts["cyber"]),
        simulate_infrastructure(counts["infrastructure"]),
        simulate_third_party(counts["third_party"]),
    ]
    df = pd.concat(frames, ignore_index=True)

    # Timestamps and censoring
    df = assign_timestamps(df)
    df = apply_censoring(df)

    # RTO reference and breach flag
    df["rto_target_hours"] = df["incident_type"].map(RTO_TARGETS)
    df["rto_met"]          = df["recovery_hours"] <= df["rto_target_hours"]

    # Severity classification
    df["severity"] = classify_severity(df["outage_hours"].values)

    # Incident ID
    df.insert(0, "incident_id", [f"INC-{i:04d}" for i in range(1, len(df) + 1)])

    # Sort by start time for realism
    df = df.sort_values("start_dt").reset_index(drop=True)
    df["incident_id"] = [f"INC-{i:04d}" for i in range(1, len(df) + 1)]

    # Column order
    col_order = [
        "incident_id", "incident_type", "severity", "start_dt",
        "outage_hours", "recovery_hours", "rto_target_hours",
        "rto_met", "censored",
    ]
    return df[col_order]


# ── summary diagnostics ───────────────────────────────────────────────────────

def print_diagnostics(df: pd.DataFrame) -> None:
    print("\n── Incident counts ──────────────────────────────────────")
    print(df["incident_type"].value_counts().to_string())

    print("\n── Recovery hours: median / 95th pct by type (uncensored only) ──")
    summary = df[~df["censored"]].groupby("incident_type")["recovery_hours"].agg(
        median=lambda x: round(x.median(), 1),
        p95=lambda x: round(x.quantile(0.95), 1),
        mean=lambda x: round(x.mean(), 1),
    )
    print(summary.to_string())

    print("\n── RTO achievement rate by type ─────────────────────────")
    rto_rate = df.groupby("incident_type")["rto_met"].mean().mul(100).round(1)
    print(rto_rate.to_string())

    print("\n── Censored observations ────────────────────────────────")
    cens = df.groupby("incident_type")["censored"].sum()
    print(cens.to_string())

    print("\n── Severity distribution ────────────────────────────────")
    print(df.groupby(["incident_type", "severity"]).size().to_string())


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = simulate_incidents(N_INCIDENTS)
    print_diagnostics(df)

    # Uncomment to persist for use in Quarto:
    df.to_csv("data/bcp_incidents.csv", index=False)
    print(f"\nSaved {len(df)} rows to data/bcp_incidents.csv")