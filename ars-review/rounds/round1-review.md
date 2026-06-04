# ARS Review — Round 1
**Date:** 2026-06-02
**Paper:** MAGNet: Multi-scale Attention Graph Network for Molecular Property Prediction
**Target:** Computational Biology and Chemistry (Elsevier, IF 3.29)

---

## Final Score: 47.37 / 100 → ❌ REJECT

## Dimension Scores (Weighted Average across 5 reviewers)

| Dimension | Weight | Score |
|-----------|--------|-------|
| Originality | 20% | 59.8 |
| Methodological Rigor | 25% | 34.4 |
| Evidence Sufficiency | 25% | 31.6 |
| Argument Coherence | 15% | 57.4 |
| Writing Quality | 15% | 69.0 |

## Reviewer Scores

| Reviewer | Originality | Methodology | Evidence | Coherence | Writing |
|----------|------------|-------------|----------|-----------|---------|
| EIC (Marchetti) | 68 | 35 | 30 | 62 | 72 |
| R1 GNN (Zhang) | 62 | 42 | 38 | 65 | 70 |
| R2 Drug (Mitchell) | 55 | 30 | 25 | 55 | 68 |
| R3 ML (Tanaka) | 64 | 40 | 45 | 60 | 70 |
| R4 DA (Chen) | 50 | 25 | 20 | 45 | 65 |

## Critical Issues (All Reviewers Agree)

1. 🔴 Synthetic data only — no MoleculeNet evaluation
2. 🔴 Trivially separable dataset (MLP 98.25%)
3. 🟡 Missing baselines: GIN, GraphSAGE, D-MPNN, SchNet, RF+Morgan
4. 🟡 No statistical rigor: single seed, no CV, no significance tests
5. 🟡 Parameter count unfairness: 261K vs 61K (4×)
6. 🟠 Placeholder ORCID, generic affiliation

## Revision Roadmap

### Priority 1 — Essential (MUST fix)
- [ ] Evaluate on MoleculeNet (BBBP, BACE, HIV, Tox21) with scaffold split
- [ ] Add baselines: GIN, GraphSAGE, D-MPNN, RF+Morgan
- [ ] 10 seeds with mean ± std
- [ ] 5-fold cross-validation
- [ ] Parameter-matched fair comparison
- [ ] Report hyperparameter search

### Priority 2 — Important (SHOULD fix)
- [ ] Statistical significance tests (paired t-test / Wilcoxon)
- [ ] Cheminformatics baselines (XGBoost + RDKit)
- [ ] Fix ablation inconsistency (3 vs 4 scales)
- [ ] Analyze 8-head instability

### Priority 3 — Recommended
- [ ] Fix ORCID and affiliation
- [ ] Add bond features
- [ ] Evaluate on larger molecules
- [ ] FLOPs/memory comparison
- [ ] Attention visualization
