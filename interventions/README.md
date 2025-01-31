# **safety_DM**

This folder contains the **custom AURA intervention** integrated into the diffusion backbones of **Stable Diffusion models**. The intervention modifies activations within the model to enhance safety and robustness.

## **Script Execution Pipeline**

The processing pipeline follows this sequence:

1. **`collect_activations.py`** – Extracts activation data from the diffusion model.
2. **`global_max.py`** – Processes and normalizes the collected activations.
3. **`auroc.py`** – Computes the AUROC (Area Under the Receiver Operating Characteristic Curve) to assess model performance.
4. **`aura_diffusion.py`** – Applies the AURA-based intervention techniques to modify activations.
5. **`generate.py`** – Generates images using the modified diffusion model.

## **Usage**

Run the scripts in the order specified above to apply the AURA intervention and generate images. Each script may have configurable parameters—refer to individual script headers for details.
