# MedMNIST v2 Benchmark Reproduction

I reproduced the ResNet-50 baselines from **MedMNIST v2** (Yang et al., *Scientific Data*, 2023) on all 18 datasets (12 2D, 6 3D), using the authors' released training code. Then I took a closer look at the datasets that didn't land within 1 AUC point of the paper.

## Summary

- **All 18 datasets, 5 runs each:** 13 of 18 mean AUCs are within 1 point of the paper. Four of the other five scored *above* the paper.
- **The five outside 1 point** were re-trained at the paper's full 100 epochs. Four of them then got 10 more runs each (15 in total), to separate real gaps from run-to-run noise.
- **BreastMNIST:** the authors' released checkpoints show it was trained at batch size 64, not the 128 the paper states. At batch 64, my 15 runs average 0.858 against the paper's 0.866 (-0.85 points).
- **VesselMNIST3D** is the one clear shortfall: 0.845 over 15 runs against the paper's 0.907. The data, code and evaluation all match the paper's version, so the cause is still open.
- **A proposal** for training a single model jointly across the 12 2D datasets, the 6 3D datasets, and all 18 together (summary below).

## Setup

| | 2D datasets | 3D datasets |
|---|---|---|
| Model | ResNet-50 at 224 px | ResNet-50+3D |
| Flags | `--model_flag resnet50 --size 28 --resize --as_rgb` | `--model_flag resnet50 --conv Conv3d --pretrained_3d i3d --as_rgb` |
| Batch size | 128 | 32 |
| Extra | | `--shape_transform` for AdrenalMNIST3D and VesselMNIST3D only, as in the paper |

- **Code:** [MedMNIST/experiments](https://github.com/MedMNIST/experiments), main branch.
- **Training:** Adam with learning rate 0.001, multiplied by 0.1 at 50% and 75% of training. Test AUC is taken from the epoch with the best validation AUC. No fixed seeds, so every run is independent.
- **Environment:** medmnist 3.0.2, ACSConv 0.1.1, current PyTorch, 2x NVIDIA RTX 3090, two runs at a time (one per GPU).
- **One code fix:** the released script masks the chosen GPU with `CUDA_VISIBLE_DEVICES` and then asks for `cuda:<original index>`, which fails for any GPU other than 0. The fix asks for `cuda:0` after masking. It only changes which GPU a run uses.
- **One deviation:** the main table uses 20 epochs instead of 100, because 100 epochs on all 18 datasets was estimated at about 10 days on this hardware. The learning-rate drops scale with the epoch count, so it's the same schedule, compressed. The datasets that missed at 20 epochs were then re-run at 100.

## Results

Gap = my mean minus the paper's value, in AUC points (AUC x 100). Negative means lower than the paper.

### 1. All 18 datasets, 5 runs each, 20 epochs ([notebook 01](notebooks/01_all_datasets_20_epochs.ipynb))

| Dataset | Paper AUC | Mine (mean ± sd) | Gap |
|---|---|---|---|
| ChestMNIST | 0.773 | 0.7764 ± 0.0031 | +0.34 |
| PathMNIST | 0.989 | 0.9848 ± 0.0023 | -0.42 |
| DermaMNIST | 0.912 | 0.9150 ± 0.0049 | +0.30 |
| OCTMNIST | 0.958 | 0.9676 ± 0.0020 | +0.96 |
| PneumoniaMNIST | 0.962 | 0.9646 ± 0.0050 | +0.26 |
| RetinaMNIST | 0.716 | 0.7170 ± 0.0052 | +0.10 |
| BreastMNIST | 0.866 | 0.8766 ± 0.0246 | **+1.06** |
| BloodMNIST | 0.997 | 0.9980 ± 0.0000 | +0.10 |
| TissueMNIST | 0.932 | 0.9300 ± 0.0009 | -0.20 |
| OrganAMNIST | 0.998 | 0.9970 ± 0.0000 | -0.10 |
| OrganCMNIST | 0.993 | 0.9934 ± 0.0008 | +0.04 |
| OrganSMNIST | 0.975 | 0.9780 ± 0.0006 | +0.30 |
| OrganMNIST3D | 0.994 | 0.9920 ± 0.0043 | -0.20 |
| NoduleMNIST3D | 0.875 | 0.8988 ± 0.0146 | **+2.38** |
| FractureMNIST3D | 0.725 | 0.7302 ± 0.0347 | +0.52 |
| AdrenalMNIST3D | 0.828 | 0.8588 ± 0.0225 | **+3.08** |
| VesselMNIST3D | 0.907 | 0.8144 ± 0.0717 | **-9.26** |
| SynapseMNIST3D | 0.851 | 0.8650 ± 0.0234 | **+1.40** |

Four of the five in bold (Breast, Adrenal, Vessel, Synapse) have only 21 to 48 minority-class cases in their validation sets, and the best epoch is picked on those sets, so checkpoint selection is noisy.

### 2. The five outside 1 point, at the paper's 100 epochs ([notebooks 02](notebooks/02_five_datasets_100_epochs.ipynb) and [03](notebooks/03_extra_runs_6_to_15.ipynb))

| Dataset | Paper AUC | Runs 1-5 | Runs 1-15 | 95% CI (15 runs) | Gap (15 runs) |
|---|---|---|---|---|---|
| BreastMNIST | 0.866 | 0.826 | 0.836 | [0.818, 0.855] | -2.97 |
| NoduleMNIST3D | 0.875 | 0.882 | | | +0.74 (5 runs) |
| AdrenalMNIST3D | 0.828 | 0.855 | 0.850 | [0.835, 0.866] | +2.23 |
| VesselMNIST3D | 0.907 | 0.814 | 0.845 | [0.822, 0.867] | -6.23 |
| SynapseMNIST3D | 0.851 | 0.861 | 0.838 | [0.806, 0.870] | -1.27 |

- **NoduleMNIST3D** came within 1 point at 100 epochs, so it didn't get extra runs. Its test set was also replaced in medmnist v2.1.0 (526 volumes then, 310 now), so the paper's number was measured on a different test set.
- **SynapseMNIST3D:** the paper's value is inside my 95% confidence interval.
- **Run-to-run noise is large on these datasets.** With identical settings, the mean of runs 6-15 differed from the mean of runs 1-5 by 1.5 (Breast), 0.6 (Adrenal), 4.5 (Vessel) and 3.4 (Synapse) points.
- The number of extra runs was fixed before any of them started, and all 15 runs are reported.

### 3. Comparison with the authors' own runs

The paper reports the mean of 3 runs per dataset. The authors also released their per-run predictions ([Zenodo record 7782114](https://zenodo.org/records/7782114)), and re-scoring them gives:

| Dataset | Authors' 3 runs | Their mean | My 15-run mean | Welch t-test p |
|---|---|---|---|---|
| BreastMNIST | 0.839, 0.906, 0.853 | 0.866 | 0.836 | 0.28 |
| AdrenalMNIST3D | 0.807, 0.852, 0.826 | 0.828 | 0.850 | 0.23 |
| VesselMNIST3D | 0.953, 0.880, 0.890 | 0.908 | 0.845 | see below |
| SynapseMNIST3D | 0.872, 0.845, 0.835 | 0.851 | 0.838 | 0.52 |

- The authors' own runs have a standard deviation of 1.9 to 4.0 AUC points, larger than the 1-point margin.
- For BreastMNIST, AdrenalMNIST3D and SynapseMNIST3D, my runs and theirs are statistically indistinguishable.
- **VesselMNIST3D is different:** 14 of my 15 runs fall below the lowest of their three (Mann-Whitney p = 0.013). I checked the paper-era data file against the current one: the train and test sets are byte-identical, and only one blank validation volume was removed. The authors' released VesselMNIST3D weights reproduce their own predictions on the current data, and the model, data loading and evaluation code match. One environment difference I haven't tested is TF32 convolution on Ampere GPUs.
- The maintainers have said the paper's numbers came from a different script than the released one, while still using the best-validation checkpoints ([issue #8](https://github.com/MedMNIST/experiments/issues/8#issuecomment-1556930350)).

### 4. BreastMNIST and batch size ([notebook 04](notebooks/04_breastmnist_batch64.ipynb))

The paper states batch size 128 for the 2D datasets. But in the authors' released ResNet-50 (224) BreastMNIST checkpoints, the BatchNorm layers counted 297, 630 and 288 batches, all multiples of 9. With 546 training images, 9 batches per epoch means a batch size of about 64. At 128 it would be 5 per epoch, and the authors' other BreastMNIST checkpoints (ResNet-18 at 224, and both models at 28 px) do show multiples of 5.

So I trained BreastMNIST 15 more times at batch size 64, with everything else unchanged. The settings and run count were fixed before the first run.

| BreastMNIST, 100 epochs | Runs | Mean | 95% CI | Gap | Runs at or above the paper |
|---|---|---|---|---|---|
| Batch 128 (paper's stated setting) | 15 | 0.836 | [0.818, 0.855] | -2.97 | 2 of 15 |
| **Batch 64 (authors' checkpoints)** | 15 | **0.858** | [0.843, 0.872] | **-0.85** | 6 of 15 |
| Authors' released runs | 3 | 0.866 | | | |

At batch 64, BreastMNIST lands within 1 point, with the paper's value inside the confidence interval. The batch-64 runs are statistically indistinguishable from the authors' runs (Welch p = 0.72). Batch 64 versus 128 gives p = 0.062, which is suggestive but not conclusive. All the other 2D results above use batch 128, as the paper states.

## Joint-training proposal (not implemented yet)

The task was to propose a way to train one model jointly across the 12 2D datasets, one across the 6 3D datasets, and one across all 18. My proposal:

- **One shared body per modality, one head per dataset.** The same ResNet-50 (2D) and ResNet-50+3D bodies as above, with a small classification head for each dataset, so no label spaces need merging. This follows Foundation Ark (Ma et al., MICCAI 2023) and Ark+ (Ma et al., *Nature*, 2025).
- **Cyclic training.** Each round visits every dataset in turn instead of mixing datasets in one batch, which Ark found hurt convergence. Each dataset's step budget per round grows with the square root of its size, with a cap, because the 2D training sets range from 546 to about 165,000 images.
- **Teacher-EMA consistency.** A teacher copy of each body, updated by exponential moving average, plus a consistency loss between student and teacher features, to limit forgetting between turns.
- **Lock-Release for the unstable datasets.** On the first visit to each of the five datasets that missed 1 point, train only its head first, then the whole network, following Foundation X (Islam et al., WACV 2025). Run with and without, as an ablation.
- **Bridging 2D and 3D for the 18-dataset model.** The 3D body is initialised by inflating the jointly trained 2D weights. Two cross-modal losses are tested as separate ablation arms: one aligns the overall 2D and 3D feature statistics, and one links the 2D organ datasets (OrganA/C/S) with OrganMNIST3D, which share the same CT source and organ labels. This part has no direct precedent in the literature, so it's the most speculative.

## Repository contents

```
notebooks/
  01_all_datasets_20_epochs.ipynb     18 datasets x 5 runs at 20 epochs
  02_five_datasets_100_epochs.ipynb   the 5 misses re-run at 100 epochs
  03_extra_runs_6_to_15.ipynb         10 more runs for the 4 remaining datasets
  04_breastmnist_batch64.ipynb        BreastMNIST at batch size 64, 15 runs
scripts/
  extract_run_results.py              reads every run's TensorBoard logs into the CSVs below
results/
  per_run_results.csv                 one row per run (168 runs)
  summary_by_dataset.csv              mean, sd and gap per dataset and setting
```

All notebooks are saved with their outputs.

## Reproducing

1. Copy the notebooks into one working folder and run notebook 01 (Run All). It downloads the authors' code, applies the GPU fix and trains every run. Notebooks 02 to 04 use the same folder and skip any run that has already finished.
2. Copy `scripts/extract_run_results.py` into that folder and run `python extract_run_results.py`. It reads the per-epoch validation and test AUC that the training script logs to TensorBoard and writes the two CSVs.

## Notes on the results files

- Test AUCs are the values written by the training script, rounded to 3 decimals. The `*_std` columns in `summary_by_dataset.csv` are population standard deviations. The confidence intervals in this README use the sample standard deviation and a t distribution.
- **ChestMNIST:** runs 1 and 2 don't have complete TensorBoard logs, so the CSVs list 3 ChestMNIST runs. The test AUCs of runs 1 and 2 (0.778 and 0.771) come from the result files the training script wrote, and notebook 01's summary includes all 5.
- **`last_*` columns:** an early version of the training script kept a reference to the model instead of a copy, so its "best" model was really the last-epoch model. I checked whether that explains the gaps. It doesn't: last-epoch scores move AdrenalMNIST3D and VesselMNIST3D further from the paper, and push ChestMNIST and TissueMNIST outside 1 point at 20 epochs.

## Reference

Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10, 41. https://doi.org/10.1038/s41597-022-01721-8
