# CLOUD

This repository contains the source code and implementation for our work, *"The Continuous Latent Ornstein-Uhlenbeck Dynamics Framework: A Scalable Latent Process Model for Multivariate Longitudinal Categorical Data."*

It provides a robust statistical framework designed for high-dimensional longitudinal data modeling, focusing primarily on continuous-time latent process models, dynamic factor analysis, and item response theory. This repository includes both the proposed CLOUD model and the baseline methods discussed in the manuscript.

## Repository Structure

The implementation is organized according to the different simulation settings detailed in the manuscript. The codebase is structured into distinct simulation modules within the `src` directory:

*   **`src/models_simu_2d/`**: Contains models for 2-dimensional simulations, utilizing non-centered parameterizations (NCP) for model fitting.
    *   `model_ncp_diag_fdt_2d.stan`: DiagOU baseline model in 2D latent space.
    *   `model_ncp_full_fdt_2d.stan`: CLOUD model in 2D latent space.
*   **`src/models_simu_4d/`**: Contains models for 4-dimensional simulations, utilizing non-centered parameterizations (NCP) for model fitting.
    *   `model_ncp_diag_fdt_4d.stan`: DiagOU baseline model in 4D latent space.
    *   `model_ncp_fix_fdt_4d.stan`: StationaryOU baseline model in 4D latent space.
    *   `model_ncp_full_fdt_4d.stan`: CLOUD model in 4D latent space.
*   **`src/models_simu_lou/`**: Contains LOU baseline simulation model.

## Technologies & Workflow

*   **Stan**: Used extensively for model estimation, evaluation, and compiling the statistical frameworks.
*   **Python**: Utilized alongside Stan for running parallel simulations and generating data visualizations.
*   **HPC / Slurm**: The models are designed for scalability, capable of utilizing Slurm batch scripts and array configurations to manage background simulation chunks across multiple nodes on high-performance computing clusters.

## Citation

For more details, please refer to our paper. If you use this code in your research, please cite our manuscript:

```bibtex
@misc{wu2026continuouslatentornsteinuhlenbeckdynamics,
      title={The Continuous Latent Ornstein-Uhlenbeck Dynamics Framework: A Scalable Latent Process Model for Multivariate Longitudinal Categorical Data}, 
      author={Zhennan Wu and Yijie Wang and Xiaoqing Huang},
      year={2026},
      eprint={2607.27520},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={[https://arxiv.org/abs/2607.27520](https://arxiv.org/abs/2607.27520)}, 
}
