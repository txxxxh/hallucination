# Top-11 hidden-delta correctness detector comparison

Dataset: 128 items, 64 correct / 64 incorrect. Evaluation uses five-fold
`StratifiedGroupKFold`, seed 42. Scaling and PCA are fit inside each training
fold. These are exploratory model-selection results, not independent-test
estimates.

| Feature set | AUROC | AUPRC | Balanced accuracy |
|---|---:|---:|---:|
| top-11 margin | 0.7866 | 0.7442 | 0.7500 |
| margin + layer-16 signed answer delta (PCA16) | 0.8391 | 0.8226 | 0.7578 |
| margin + layer-16 positive/negative delta (PCA8 each) | 0.8381 | 0.8498 | 0.7422 |
| margin + layer-16 original + positive/negative delta (PCA8 each) | 0.8518 | 0.8587 | 0.7344 |
| same, PCA12 each | **0.8594** | **0.8648** | 0.7422 |
| same, PCA16 each | 0.8555 | 0.8647 | 0.7266 |
| separately aggregated masked-positive/masked-negative states | 0.8477 | 0.8599 | 0.7266 |
| layer-16 and layer-24 positive/negative delta | 0.8271 | 0.8218 | 0.7656 |
| answer-direction projection + margin (LR) | 0.7930 | 0.7829 | 0.7109 |
| input-embedding signed delta | 0.7434 | 0.7578 | 0.6562 |
| input-embedding signed delta + margin | 0.8003 | 0.7872 | 0.7500 |
| hidden-change spectrum (best layer) | 0.7026 | 0.7059 | 0.6406 |
| Hessian diagonal curvature | 0.7234 | 0.6943 | 0.7031 |
| Hessian diagonal curvature + margin | 0.7603 | 0.7223 | 0.7031 |
| token-level response + best hidden model | 0.8152 | 0.7915 | 0.7422 |
| NMS spans + original/positive/negative delta | 0.8357 | 0.8371 | 0.7734 |
| embedding-MMR spans + original/positive/negative delta | 0.8401 | 0.8374 | 0.7500 |

Nonlinear classifiers did not improve the best representation: RBF-SVM
reached 0.8074 AUROC at best and shallow histogram gradient boosting reached
0.7893. Low-dimensional layer-trajectory features also reduced AUROC to
0.8171.

## Recommended fixed configuration

Use the original absolute-margin top-11 spans and layer 16 at the last answer
token. Construct three separate blocks:

1. original hidden state;
2. positive-margin-weighted hidden delta;
3. negative-margin-magnitude-weighted hidden delta.

Fit a separate StandardScaler + PCA12 to each hidden block inside the training
fold. Standardize the 39 margin features separately, concatenate the four
blocks, and fit balanced logistic regression with C=0.5.

The next valid measurement should freeze this configuration and evaluate it on
new grouped data. Selecting PCA size, layer, or representation again on the
same 128 items would increase selection bias.
