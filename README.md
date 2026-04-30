# When the Clock Starts Ticking: A Survival Analysis of Business Continuity Recovery Times

> Quantifying RTO achievement across information security, infrastructure, and third-party incidents

**Author:** Patrick Lefler </br>
**Published:** April 14, 2026 <br/>
**Rendered:** https://patrick-lefler.github.io/pyQuarto_business_continuity_recovery_time_analysis_2026-04-14/

---

## Project Introduction

> Applies Kaplan-Meier survival analysis to 800 simulated business continuity incidents to quantify where RTO commitments break down, by how much, and why mean-based reporting conceals the answer.

---

## Overview

Most business continuity programs report recovery performance as a mean recovery time measured against a stated RTO. That framing systematically misrepresents the underlying distribution: a small number of severe, long-duration incidents inflate the mean while concealing tail risk entirely. This project applies the Kaplan-Meier estimator — a non-parametric survival analysis method — to 800 synthetic incidents across three categories (cyber, infrastructure, third-party) to produce empirical recovery time distributions that account for right-censored observations.

The dataset is calibrated to published industry benchmarks from the Uptime Institute Annual Outage Analysis and ENISA Threat Landscape reports, with incident durations linked via a Gaussian copula (ρ = 0.65) to model realistic co-movement between outage severity and recovery time. The intended output is a decision-support framework for risk committees and operational resilience teams who need defensible, quantitative evidence for RTO target-setting — not achievement percentages against targets set without reference to the empirical distribution.

---

## Tech Stack

- **Language:** Python
- **Framework:** [Quarto](https://quarto.org/)
- **Primary Libraries:** pandas, numpy, scipy, lifelines, plotly, great_tables
- **Deployment/Output:** Embedded-resource HTML Document

---

## Repository Structure

```
├── data/               # Raw and processed data (ensure .gitignore for sensitive info)
├── scripts/            # Helper Python scripts or modules
├── models/             # Saved model objects (.pkl)
├── output/             # Rendered HTML files
├── _brand.yml          # Brand configuration (colors, typography)
├── _quarto.yml         # Project configuration
└── index.qmd           # Main Quarto entry point
```

---

## Key Findings

**Infrastructure targets are miscalibrated, not misexecuted.** The 8-hour RTO is breached roughly half the time despite infrastructure incidents carrying the shortest median recovery times of any category. The failure is in target-setting: any incident requiring physical intervention, vendor dispatch, or multi-system coordination will breach the target regardless of response team performance. Recalibrating to 12 hours for standard incidents replaces an aspirational number with a defensible one.

**Information security tail risk is invisible in mean-based reporting.** The p95 recovery time for cyber incidents approaches 180 hours — one in twenty incidents remains in active recovery after a full calendar week. A mean recovery time of ~50 hours, reported against a 24-hour RTO, obscures this entirely. The Kaplan-Meier curve reveals a structural step change around 48 hours that marks the boundary between incidents following the standard escalation path and those requiring a fundamentally different response posture.

**Third-party RTO governance requires contractual separation of obligations.** A meaningful portion of third-party recovery time is constitutionally outside the firm's control: notification lag, incident scoping, and minimum vendor response time precede any internal action. A single RTO measured from incident start conflates two distinct obligations. Separating the vendor notification window (enforceable by contract) from the firm's post-notification recovery window (addressable through internal process) creates two measurable, enforceable targets where currently there is one unenforceable aggregate.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

Patrick Lefler — [LinkedIn](https://www.linkedin.com/in/patricklefler/) | [Website](https://patrick-lefler.github.io) | [Substack](https://substack.com/@pflefler)

