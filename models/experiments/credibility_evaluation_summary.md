# AgentPulse Behavioral ML Credibility Evaluation Report (Synthetic Data)

> **Note**: All generated datasets and evaluations in this report use synthetic agent session simulations. Results measure evaluation methodology sensitivity (label leakage, template cloning, and row-level vs scenario-grouped splitting) and must NOT be interpreted as real-world attack detection effectiveness.

## 1. Read-Only Audit of Existing `data/behavioral_dataset.csv` (200 Rows)
- **Total Rows**: 200 (`NORMAL`: 100, `SUSPICIOUS`: 100)
- **Has Scenario-Family Column**: `False`
- **Unique Discrete 8-Feature Patterns**: 21 across 200 rows
- **Rule `security_detection_count > 0` Accuracy**: 100.00%
- **Single-Feature 5-Fold Decision Stump (`security_detection_count`)**: 100.00%
- **Single-Feature 5-Fold Decision Stump (`tool_result_count`)**: 86.50%
- **Single-Feature 5-Fold Decision Stump (`total_events`)**: 61.00%

## 2. Benchmark A: Exact 152 Synthetic Sessions (`expand_dataset.py` Templates)
- **Samples**: 152 (`NORMAL`: 74, `SUSPICIOUS`: 78)
- **Mean Security Event Overhead (`total_events - non_security_total_events`)**: NORMAL = `0.0`, SUSPICIOUS = `1.7692`

| Feature Set | Split Mode | Holdout Acc | Holdout Macro-F1 | CV Method (Folds) | CV Mean Acc (±Std) | CV Mean Macro-F1 | Pooled OOF Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `11_with_detector_leakage` | **Row-Level** (Leaks templates) | 1.0000 | 1.0000 | `StratifiedKFold` (5) | 1.0000 (±0.0000) | 1.0000 | `[[74, 0], [0, 78]]` |
| `11_with_detector_leakage` | **Scenario-Grouped** (0 overlap) | 0.8333 | 0.8194 | `StratifiedGroupKFold` (3) | **0.5105** (±0.0753) | **0.3612** | `[[74, 0], [75, 3]]` |
| `9_raw_total_events_indirect_leakage` | **Row-Level** (Leaks templates) | 1.0000 | 1.0000 | `StratifiedKFold` (5) | 1.0000 (±0.0000) | 1.0000 | `[[74, 0], [0, 78]]` |
| `9_raw_total_events_indirect_leakage` | **Scenario-Grouped** (0 overlap) | 0.2778 | 0.2174 | `StratifiedGroupKFold` (3) | **0.5543** (±0.1811) | **0.4688** | `[[65, 9], [60, 18]]` |
| `9_deleaked_behavioral` | **Row-Level** (Leaks templates) | 0.9355 | 0.9354 | `StratifiedKFold` (5) | 0.8750 (±0.0383) | 0.8735 | `[[63, 11], [8, 70]]` |
| `9_deleaked_behavioral` | **Scenario-Grouped** (0 overlap) | 0.5741 | 0.4014 | `StratifiedGroupKFold` (3) | **0.5543** (±0.1811) | **0.4688** | `[[65, 9], [60, 18]]` |
| `8_pure_tool_timing` | **Row-Level** (Leaks templates) | 0.9355 | 0.9354 | `StratifiedKFold` (5) | 0.8750 (±0.0383) | 0.8735 | `[[63, 11], [8, 70]]` |
| `8_pure_tool_timing` | **Scenario-Grouped** (0 overlap) | 0.2778 | 0.2174 | `StratifiedGroupKFold` (3) | **0.5543** (±0.1811) | **0.4688** | `[[65, 9], [60, 18]]` |

- **Group CV Class-Coverage Check**: Requested 5-fold StratifiedGroupKFold cannot place SUSPICIOUS sessions in all 5 validation folds because dataset has 6 NORMAL families and 3 SUSPICIOUS families. Automatically used 3-fold StratifiedGroupKFold (max folds with full two-class coverage).

## 3. Benchmark B: Enriched 240 Synthetic Sessions (12 Scenario Families with Hard Negatives)
- **Samples**: 240 (`NORMAL`: 120, `SUSPICIOUS`: 120)
- **Mean Security Event Overhead (`total_events - non_security_total_events`)**: NORMAL = `2.7333`, SUSPICIOUS = `1.5833`

| Feature Set | Split Mode | Holdout Acc | Holdout Macro-F1 | CV Method (Folds) | CV Mean Acc (±Std) | CV Mean Macro-F1 | Pooled OOF Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `11_with_detector_leakage` | **Row-Level** (Leaks templates) | 1.0000 | 1.0000 | `StratifiedKFold` (5) | 1.0000 (±0.0000) | 1.0000 | `[[120, 0], [0, 120]]` |
| `11_with_detector_leakage` | **Scenario-Grouped** (0 overlap) | 1.0000 | 1.0000 | `StratifiedGroupKFold` (5) | **0.9333** (±0.1333) | **0.8800** | `[[120, 0], [20, 100]]` |
| `9_raw_total_events_indirect_leakage` | **Row-Level** (Leaks templates) | 0.9375 | 0.9375 | `StratifiedKFold` (5) | 0.9542 (±0.0276) | 0.9541 | `[[113, 7], [4, 116]]` |
| `9_raw_total_events_indirect_leakage` | **Scenario-Grouped** (0 overlap) | 0.6667 | 0.6667 | `StratifiedGroupKFold` (5) | **0.6517** (±0.2893) | **0.5763** | `[[77, 43], [40, 80]]` |
| `9_deleaked_behavioral` | **Row-Level** (Leaks templates) | 0.8542 | 0.8541 | `StratifiedKFold` (5) | 0.8750 (±0.0437) | 0.8749 | `[[104, 16], [14, 106]]` |
| `9_deleaked_behavioral` | **Scenario-Grouped** (0 overlap) | 0.6667 | 0.6667 | `StratifiedGroupKFold` (5) | **0.6517** (±0.2893) | **0.5763** | `[[77, 43], [40, 80]]` |
| `8_pure_tool_timing` | **Row-Level** (Leaks templates) | 0.8542 | 0.8541 | `StratifiedKFold` (5) | 0.8750 (±0.0437) | 0.8749 | `[[104, 16], [14, 106]]` |
| `8_pure_tool_timing` | **Scenario-Grouped** (0 overlap) | 0.6667 | 0.6667 | `StratifiedGroupKFold` (5) | **0.7017** (±0.2802) | **0.6563** | `[[87, 33], [40, 80]]` |

- **Group CV Class-Coverage Check**: All 5 grouped folds contain both NORMAL and SUSPICIOUS classes with 0 group overlap.

