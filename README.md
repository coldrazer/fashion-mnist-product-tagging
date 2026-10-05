# Automated Product Image Tagging with Deep Learning

Classifying Fashion-MNIST product photos: three deep learning models compared, with a deployment recommendation based on a confidence threshold operating analysis.

## Headline results

* Recommended model (regularized CNN): **94.10% test accuracy**, macro F1 0.941 on 10,000 held out images
* **48% fewer tagging errors** than the fully connected baseline (88.63%)
* Business operating point: at confidence >= 0.95, **79.7% of images auto tag at 99.25% accuracy**; the rest go to a human reviewer

## Approach

Three architectures trained with the identical procedure (Adam, cross entropy, batch 128, early stopping on validation loss), so differences come from architecture alone:

| Model | Params | Val accuracy | Overfitting gap |
|---|---|---|---|
| MLP baseline | 235K | 89.54% | 2.11 pts |
| Basic CNN | 422K | 92.69% | 3.91 pts |
| Regularized CNN | 872K | 94.93% | 1.61 pts |

The regularized CNN adds batch normalization, dropout, data augmentation (random shifts), and a learning rate schedule. The basic CNN's validation loss rises while training loss falls (classic overfitting); the regularized model's curves stay aligned through epoch 39 of 40.

Key findings: convolution is the single biggest gain (spatial structure matters); 83% of all test errors fall below 0.9 confidence, so confidence is a reliable control knob; errors concentrate where the data exploration predicted, among visually similar upper body garments (Shirt is hardest, F1 0.81).

## Operating thresholds

| Confidence threshold | Auto tagged | Accuracy on auto tagged | Sent to human |
|---|---|---|---|
| 0.90 | 84.5% | 98.80% | 1,551 |
| 0.95 | 79.7% | 99.25% | 2,032 |
| 0.99 | 69.1% | 99.70% | 3,090 |

Suggested starting point: 0.95. A few errors are confidently wrong, so periodic human spot checks remain worthwhile.

## Limitations

Fashion-MNIST is 28x28 grayscale and perfectly centered; real catalog photos need revalidation. No systematic hyperparameter search was run. Softmax confidence is uncalibrated (temperature scaling recommended before production).

## Contents

* `fashion_mnist_deep_learning.ipynb` — the full analysis, end to end
* `report.pdf` — the complete writeup with figures, confusion matrix, and threshold analysis
* Run it on Kaggle: [Fashion-MNIST: MLP vs CNN vs Regularized CNN](https://www.kaggle.com/code/coldrazer/fashion-mnist-mlp-vs-cnn-vs-regularized-cnn)
* Dataset and trained models on Kaggle: [Fashion-MNIST CNN Models and Results](https://www.kaggle.com/datasets/coldrazer/fashion-mnist-cnn-models-and-results)

## Reproduce

```bash
pip install -r requirements.txt
jupyter notebook fashion_mnist_deep_learning.ipynb
```

Environment: Python 3.12, Keras 3.15.1 (torch backend), scikit-learn, pandas, matplotlib, seaborn. Trained on a single GPU. Random seed 42; GPU kernels are not fully deterministic so reruns may differ in the last decimals.

## Author

Christopher Guerrero, October 2026. Deep learning coursework project.
