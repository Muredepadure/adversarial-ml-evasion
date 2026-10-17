# Adversarial ML: Evasion Attacks on an MNIST Classifier

A hands-on study of **white-box and black-box evasion attacks** against a CNN
image classifier. Implements PGD (Projected Gradient Descent) from scratch in
PyTorch, reproduces it with IBM's Adversarial Robustness Toolbox (ART), and runs
HopSkipJump (HSJ) — a decision-based black-box attack — against the same model.

## Overview

Evasion attacks perturb an input at **inference time** to make a trained model
produce a wrong output, while the input looks unchanged to a human. This project
demonstrates two ends of the spectrum:

- **PGD** — a white-box attack that uses the model's gradients directly.
- **HopSkipJump** — a black-box attack that sees only the model's output label
  and estimates its way to the decision boundary.

## What this project does

1. Trains a CNN on MNIST (~98.5% clean test accuracy)
2. Implements PGD from scratch in PyTorch (iterative gradient ascent on the
   input, projected into an L∞ ε-ball)
3. Reproduces the attack with ART to validate the from-scratch version
4. Runs HopSkipJump (black-box) against the same model
5. Compares the attacks by success rate, perturbation budget, and query cost

## Key results

- Clean test accuracy: **~98.5%**
- Under PGD (L∞, ε=0.3), accuracy dropped to **~0%** — the model is wrong on
  essentially every perturbed image, while the perturbations stay visually
  imperceptible
- PGD (white-box) reaches this in ~40 gradient steps; HopSkipJump (black-box)
  achieves comparable misclassification using only output labels, at the cost of
  far more model queries

<!-- Add figures here once exported, e.g.:
![Adversarial example](images/adversarial_example.png)
![Robustness curve: accuracy vs epsilon](images/robustness_curve.png)
-->

## What I learned

- **White-box vs black-box** is fundamentally a trade-off between *gradient
  access* and *query budget*. PGD is cheap but assumes full model access; HSJ is
  realistic against a deployed API but spends thousands of queries.
- **ε is a threat-model choice, not a fixed value.** It defines how much
  perturbation counts as "imperceptible" or realistic, and it's meaningless
  without a norm (L∞ caps every pixel; L2 caps overall change).
- **Robustness is a curve, not a single number** — the honest way to report it
  is the smallest ε that breaks the model, across a sweep.

## Limitations

MNIST is a toy benchmark, chosen here for fast iteration while learning the
mechanics. Results on it don't automatically transfer to harder datasets; a real
robustness claim would need validation on CIFAR-10 or ImageNet.

## Stack

Python · PyTorch · Adversarial Robustness Toolbox (ART) · Google Colab

## Running it

Open the notebook in Google Colab (GPU runtime recommended) and run all cells top
to bottom. Dependencies install in the first cell.
