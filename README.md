
This is the official repository for the code associated with the following paper:

$\alpha$-TCVAE: On The Relationshipo Between Disentanglement and Diversity (https://openreview.net/pdf?id=ptXo0epLQo) **Published as a conference paper at ICLR 2024**.

This repository contains a PyTorch-based framework and benchmarking suite to facilitate research on methods for learning disentangled representations and understanding their relationship with generative diversity. It includes all the code, benchmarks, and method implementations used in the $\alpha$-TCVAE paper.

**Table of Contents**:

- [$\alpha$-TCVAE: ON THE RELATIONSHIP BETWEEN DISENTANGLEMENT AND DIVERSITY](#-tcvae-on-the-relationship-between-disentanglement-and-diversity)
  - [Quick Overview](#quick-overview)
  - [Installation and Requirements](#installation-and-requirements)
  - [Implemented Methods and Metrics](#implemented-methods-and-metrics)
    - [Training Objectives and Architectures](#training-objectives-and-architectures)
    - [Metrics](#metrics)
    - [Benchmarks](#benchmarks)
  - [Training](#training)
  - [Evaluation & Visualization](#evaluation--visualization)
  - [Outputs](#outputs)
  - [Complete Examples](#complete-examples)
    - [Training and Evaluating $\alpha$-TCVAE](#training-and-evaluating--tcvae)
  - [Citation](#citation)

---

## Installation and Requirements

This repository was tested and evaluated using standard deep learning libraries. While specific versions are not listed in the paper, a common setup would include:

- Python (e.g., 3.8+)
- PyTorch (e.g., >=1.10 on GPU)

A suitable Conda environment can typically be set up by running `conda env create -f environment.yaml` (assuming an `environment.yaml` file is provided in the [official repository](https://github.com/Cmeo97/Alpha-TCVAE)).

If you wish to utilize logging with Weights & Biases, ensure you have an account and configure your key as needed.

---

## Implemented Methods and Metrics

This repository contains code to train and evaluate various VAE models, focusing on disentanglement and generative diversity.

### Training Objectives and Architectures

The primary contribution is **$\alpha$-TCVAE**, a VAE optimized using a novel convex lower bound of the joint total correlation (TC). This bound generalizes the $\beta$-VAE lower bound and can be reduced to a convex combination of Variational Information Bottleneck (VIB) and Conditional Entropy Bottleneck (CEB) terms.

The paper compares $\alpha$-TCVAE against:
-   **Vanilla VAE** (Kingma & Welling, 2013)
-   **$\beta$-VAE** (Higgins et al., 2016) 
-   **FactorVAE** (Kim & Mnih, 2018) 
-   **$\beta$-TCVAE** (Chen et al., 2018) 
-   **B-VAE+HFS** (Roth et al., 2023) 
-   **StyleGAN** (Karras et al., 2019) for diversity and visual fidelity comparison.

Architectural details for encoders and decoders are adopted from Roth et al. (2023). Hyperparameters used are detailed in Table 2 of the paper.

### Metrics

The following metrics were used for evaluation:
-   **Disentanglement Metrics**:
    -   **DCI (Disentanglement, Completeness, Informativeness)** (Eastwood & Williams, 2018) 
    -   **SNC (Single Neuron Classification)** (Mahon et al., 2023) 
    -   **MIG (Mutual Information Gap)** is mentioned in Appendix D, Figure 12.
    -   **Unfairness** (Locatello et al., 2019a) 
-   **Generative Quality & Diversity Metrics**:
    -   **FID (Fréchet Inception Distance)** (Heusel et al., 2017) to measure image faithfulness.
    -   **Vendi Score** (Friedman & Dieng, 2022) to measure diversity of sampled images.
-   **Downstream Task Performance**:
    -   **Attribute Classification Accuracy** using MLP on latent representations.
    -   **Reinforcement Learning (RL) Performance** in the Loconav Ant Maze task using the Director agent (Hafner et al., 2022).

### Benchmarks

The models were trained and evaluated on the following datasets:
-   **3DShapes** (Burgess & Kim, 2018) 
-   **MPI3D-Real** (Gondal et al., 2019) - noted as the most realistic factorized dataset in the study.
-   **Cars3D** (Reed et al., 2015) 
-   **Teapots** (Moreno et al., 2016) 
-   **CelebA** (Liu et al., 2015) - noted as the most realistic and complex dataset overall.

All images were $64 \times 64$ RGB. Datasets are typically split with 80% for training and 20% for evaluation, with training performed in a fully unsupervised way.

---

## Training

Training involves optimizing the VAE models using their respective objective functions. For $\alpha$-TCVAE, this means maximizing the novel TC lower bound defined in Equation 6 of the paper. Key hyperparameters like $\alpha$, $\beta$ (implicitly, as $\alpha$-TCVAE generalizes $\beta$-VAE where for $\alpha=0$ it reduces to $\beta$-VAE ), learning rate, and latent dimensions vary per dataset and are detailed in Table 2 and Appendix B.

Training epochs are typically set to 50 for the datasets used.

Refer to the [official repository](https://github.com/Cmeo97/Alpha-TCVAE) for specific training scripts and command-line arguments.

---

## Evaluation & Visualization

Evaluation is performed on trained models using the metrics listed above.
-   For **generative quality and diversity**, images are generated using two strategies:
    1.  Sampling noise from a multivariate standard normal and decoding it.
    2.  Encoding an image, performing latent traversals (adjusting a chosen dimension by +/- standard deviations), and decoding.
-   **Disentanglement metrics** (DCI, SNC) are computed on the learned latent representations.
-   **Downstream tasks** like attribute classification and RL performance are also evaluated.

Visualizations include:
-   **Latent traversals** (Figure 1, Figure 16) to qualitatively assess disentanglement and visual fidelity.
-   **Ground truth vs. reconstructions** (Figure 1).
-   **Correlation matrices** for different metrics (Figure 7, Figure 10).
-   **Sensitivity analysis plots** for the $\alpha$ hyperparameter (Figures 17, 18, 19, 20 in Appendix H).

---

## Complete Examples

### Training and Evaluating $\alpha$-TCVAE

To train and evaluate the $\alpha$-TCVAE model as presented in the paper, you would typically:
1.  **Set up the environment**: Clone the [official repository](https://github.com/Cmeo97/Alpha-TCVAE) and install necessary dependencies.
2.  **Prepare the dataset**: Ensure the desired dataset (e.g., MPI3D-Real) is available and correctly formatted.
3.  **Run the training script**: Use the provided scripts in the repository, specifying $\alpha$-TCVAE as the model and configuring hyperparameters according to the paper (e.g., from Table 2 and Appendix B for MPI3D-Real: $\alpha=0.25$, $\beta=5$ (this refers to the $\beta$ in $\beta$-VAE, $\alpha$-TCVAE has its own $\alpha$ hyperparameter for the convex combination ), latent dim K=10).
    ```bash
    # Example 
    python train.py --model alpha_tcvae --dataset mpi3d_real --alpha_param 0.25 --latent_dim 10 --epochs 50 
    ```
    (Note: The paper uses $\alpha$ as the hyperparameter trading off VIB and CEB terms for $\alpha$-TCVAE, and separately refers to a $\beta$ parameter for the $\beta$-VAE and other baselines, often used for the KL divergence weight. For $\alpha$-TCVAE, the value $\alpha=0.25$ is used for experiments in the main paper).
4.  **Run evaluation scripts**: After training, use scripts to compute the relevant metrics (FID, Vendi, DCI, SNC, etc.) and generate visualizations.
    ```bash
    # Example 
    python evaluate.py --model_checkpoint <path_to_checkpoint> --dataset mpi3d_real --metrics all --visualize_traversals
    ```

---

## Citation

If you find this work useful, please consider citing it:

[$\alpha$-TCVAE: ON THE RELATIONSHIP BETWEEN DISENTANGLEMENT AND DIVERSITY](https://openreview.net/forum?id=OKcJhpQiGiX) , **Cristian Meo, Louis Mahon, Anirudh Goyal, Justin Dauwels,** *Published as a conference paper at ICLR 2024*.

```bibtex
@inproceedings{
meo2024alpha,
title={{\alpha}-TCVAE: ON THE RELATIONSHIP BETWEEN DISENTANGLEMENT AND DIVERSITY},
author={Cristian Meo and Louis Mahon and Anirudh Goyal and Justin Dauwels},
booktitle={International Conference on Learning Representations (ICLR)},
year={2024},
url={[https://openreview.net/forum?id=YOUR_PAPER_ID_HERE](https://openreview.net/forum?id=YOUR_PAPER_ID_HERE)} // Placeholder: Update with actual OpenReview ID or DOI
}
