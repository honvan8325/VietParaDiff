# Training curriculum

1. **VAE stage**: train `DualBandVAE` with pixel, frequency, KL, and low/high decorrelation losses.
2. **HTR stage**: train line-wise CTC recognizer with per-line crops.
3. **Topology stage**: train diacritic heatmap detector using exact grapheme targets when available and weak layout targets otherwise.
4. **Style-layout stage**: train content, factorized style, and autoregressive layout planner with teacher forcing decay.
5. **Diffusion stage**: freeze dependencies, train U-Net and refiner on low/high latent targets.

Default config is intentionally sized for serious experiments, not for toy CPU demos. Use `vpd-smoke` or CLI overrides for tiny local checks.
