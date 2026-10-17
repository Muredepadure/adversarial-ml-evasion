# Adversarial ML: Evasion Attacks on an MNIST Classifier

A hands-on study of **white-box and black-box evasion attacks** against a CNN
image classifier. Implements PGD (Projected Gradient Descent) from scratch in
PyTorch, reproduces it with IBM's Adversarial Robustness Toolbox (ART), and runs
HopSkipJump (HSJ) — a decision-based black-box attack — against the same model.

## Overview

Evasion attacks perturb an input at **inference time** to make a trained model
produce a wrong output, while the input stays close to the original. This project
contrasts two ends of the attacker-knowledge spectrum:

- **PGD** — white-box: uses the model's gradients directly.
- **HopSkipJump** — black-box: sees only the model's output label and estimates
  its way to the decision boundary using many queries.

## What this project does

1. Trains a CNN on MNIST (98.5% clean test accuracy)
2. Implements PGD from scratch in PyTorch (iterative gradient ascent on the
   input, projected into an L∞ ε-ball)
3. Reproduces the attack with ART to validate the from-scratch version
4. Runs HopSkipJump (black-box) against the same model
5. Compares the two attacks by access required, success, perturbation size, and
   query cost

## Key results

- Clean test accuracy: **98.5%**
- **PGD (white-box, L∞ ε=0.3):** drove batch accuracy from ~99% to **0%** in ~40
  gradient steps
- **HopSkipJump (black-box):** fooled **5/5** test images using only output
  labels

Head-to-head on the same 5 images:

| Metric | PGD (white-box) | HSJ (black-box) |
|---|---|---|
| Model access | Gradients | Labels only |
| Images fooled | 5/5 | 5/5 |
| Avg L2 perturbation | 5.49 | 2.07 |
| Queries per image | ~40 | ~1000s |

![Clean vs HSJ adversarial](images/hsj_comparison.png)

## What I learned

- **White-box vs black-box is a trade-off between gradient access and query
  budget.** PGD is cheap but assumes full access; HSJ is realistic against a
  deployed API but spends thousands of queries per image.
- **Less access can still mean a subtler attack.** HSJ produced *smaller*
  perturbations than PGD (L2 2.07 vs 5.49) because it minimizes distance to the
  decision boundary, whereas PGD fills its entire ε budget regardless. Access
  and stealth are independent axes.
- **ε is a threat-model choice, not a fixed value**, and only meaningful with a
  norm (L∞ caps every pixel; L2 caps overall change). Robustness is a curve, not
  a single number.

## Limitations

MNIST is a toy benchmark, chosen for fast iteration while learning the mechanics.
Results don't automatically transfer to harder datasets; a real robustness claim
would need validation on CIFAR-10 or ImageNet.

## Stack

Python · PyTorch · Adversarial Robustness Toolbox (ART) · Google Colab

## Running it

Open the notebook in Google Colab (GPU runtime recommended) and run all cells top
to bottom. Keep the pipeline in float32 (the model is not converted to double) so
it stays compatible with ART.
