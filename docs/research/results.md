# Results

The ensemble was evaluated on two public benchmarks. Fold/architecture selection used validation EMA
`fg_dice`; the test sets were never used for selection.

## Headline

- **MSLesSeg-2024 (DS001):** 0.7179 Dice — exceeding the ICPR 2024 MadSeg challenge winner (0.714).
- **WMH-2017 (DS002):** 0.803 Dice.

## Per-dataset summary

| Dataset | Cases | Dice |
|---------|-------|------|
| MSLesSeg-2024 (DS001) | 93 | 0.718 |
| WMH-2017 (DS002) | 60 | 0.803 |

## What drives the accuracy

- **Architecture diversity** — CNN 3D, ResEncL 3D, and 2.5D make different errors; averaging their softmax
  outputs is consistently stronger than any single model.
- **Fold diversity** — combining the best folds per architecture adds further robustness.
- **Validation-only selection** — the recommended Best-2/arch ensemble is chosen on validation metrics, so the
  reported test numbers are not inflated by test-set leakage.

See [reproducing.md](reproducing.md) for how the evaluation was run.
