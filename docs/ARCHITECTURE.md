# Architecture

VietParaDiff factorizes one-shot paragraph handwriting generation into content,
style, layout, and dual-band latent synthesis.

```text
Target text T -> VisualGraphemeContentEncoder -> G(T)
Reference R   -> FactorizedStyleEncoder       -> S_global, S_layout, S_stroke
G(T), S       -> AutoregressiveLayoutPlanner  -> boxes, anchors, break logits
boxes/anchors -> rasterize_layout             -> 5-channel soft layout fields
image X       -> DualBandVAE                  -> low_z, high_z
low_z + noise -> ConditionalUNet              -> predicted noise
low_z + layout + S_stroke -> Refiner          -> high_z
low_z + high_z -> VAE decoder                 -> generated paragraph image
```

The source intentionally uses explicit stage methods (`forward_vae`, `forward_htr`,
`forward_topology`, `forward_style_layout`, `forward_diffusion`, `generate`) to avoid
training with random dependencies.
