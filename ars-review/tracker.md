# ARS Iteration Tracker

| Round | Score | Decision | Key Change | Date |
|-------|-------|----------|------------|------|
| R1 | 47.37 | Reject | Initial submission — synthetic data only | 2026-06-02 |

## Target: Score ≥ 80 (Accept)

## Gap Analysis
- Current: 47.37
- Target: 80.00
- Gap: 32.63 points

## Dimension Targets (to reach 80)
| Dimension | R1 Score | Weight | R1 Weighted | Need | Strategy |
|-----------|----------|--------|-------------|------|----------|
| Originality | 59.8 | 20% | 11.96 | 15.2 | Threshold function comparison (soft vs hard vs garrote) |
| Methodology | 34.4 | 25% | 8.56 | 20.0 | MoleculeNet + scaffold split + 10 seeds + CV |
| Evidence | 31.6 | 25% | 7.88 | 20.0 | Real benchmarks + 6 baselines + significance tests |
| Coherence | 57.4 | 15% | 8.69 | 12.0 | Fix ablation inconsistency + honest framing |
| Writing | 69.0 | 15% | 10.28 | 12.75 | Minor polish |

## Round 1 Fixes (Priority Order)
1. Run MoleculeNet benchmarks (BBBP, BACE, HIV, Tox21)
2. Add GIN, GraphSAGE, D-MPNN, SchNet baselines
3. Add RF + Morgan fingerprints baseline
4. 10 seeds, mean ± std
5. 5-fold cross-validation
6. Parameter-matched comparison
7. Hyperparameter search
8. Statistical significance tests
9. Fix ORCID
10. Fix ablation inconsistency
