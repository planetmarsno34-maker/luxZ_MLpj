# Lux DNA → Stable Diffusion

A generative pipeline that translates real bacterial DNA into images and animations using Stable Diffusion. Every visual output — colour, brightness, and structure — is determined directly by measurable properties of the DNA.

The subject is _Vibrio fischeri_ ES114, a bioluminescent marine bacterium whose _lux_ operon is the canonical model for quorum sensing. The project also introduces **LuxZ**, a synthetic gene that does not exist in nature, constructed from first principles within the same biological coordinate system as the real genes.

---

## What it does

```
Real DNA from NCBI
      ↓
Extract biological features (GC content · k-mer fingerprint · lux box score)
      ↓
Build a gene map in latent space (PCA · ESM2)
      ↓
Invent LuxZ ★ — place it · give it a sequence · fold it · decode its features
      ↓
Convert features → Stable Diffusion prompts (one per gene)
      ↓
Generate and composite images (txt2img · img2img · ControlNet · Lab blend)
      ↓
Animate: gene identity morph · organism video · bioluminescence activation
```

Each gene produces its own image. The colour (cold blue-white vs warm amber-green) comes from GC content. The brightness comes from the lux box score. The structure comes from the ESMFold-predicted protein backbone fed through ControlNet.

## Genes

| Gene       | Role                                            | Lux box score |
| ---------- | ----------------------------------------------- | ------------- |
| luxA       | Luciferase α-subunit — produces light           | 0.500         |
| luxB       | Luciferase β-subunit — produces light           | 0.500         |
| luxC       | Acyl reductase — builds aldehyde fuel           | 0.600         |
| luxD       | Acyl transferase — builds aldehyde fuel         | 0.500         |
| luxE       | Acyl-protein synthetase — builds aldehyde fuel  | 0.600         |
| luxG       | Flavin reductase — provides FMNH₂ to luciferase | 0.550         |
| luxI       | Autoinducer synthase — quorum sensing signal    | 0.900         |
| luxR       | Transcription factor — activates lux operon     | 0.800         |
| luxT       | Regulator (indirect)                            | 0.550         |
| qsrP       | Quorum sensing related (indirect)               | 0.650         |
| **luxZ** ★ | **Synthetic hybrid gene**                       | 0.620         |

Source: _Vibrio fischeri_ ES114, chromosome 2, NCBI accession `CP000021.2`

---

## Setup

```bash
pip install biopython scikit-learn scikit-image matplotlib numpy \
            diffusers transformers accelerate einops sentencepiece
```

Also requires **ffmpeg** (for Step 8e video processing):

```bash
# macOS
brew install ffmpeg
# or via conda
conda install -c conda-forge ffmpeg
```

Models download automatically on first run:

- Stable Diffusion v1.5 — `runwayml/stable-diffusion-v1-5` (~4 GB)
- ESM2 — `facebook/esm2_t6_8M_UR50D`
- ESMFold (~2.7 GB)
- ControlNet scribble — `lllyasviel/control_v11p_sd15_scribble`

Compute backends are detected automatically:

| Backend | Hardware              | Precision |
| ------- | --------------------- | --------- |
| `cuda`  | NVIDIA GPU            | float16   |
| `mps`   | Apple Silicon (M1–M4) | float32   |
| `cpu`   | Any machine           | float32   |

Cell 0 detects it is not on Colab and skips setup automatically.

## Media Files

TD input image and video are available here:
[Google Drive](https://drive.google.com/drive/folders/1y10HSUoeiShwtrbsiKCebK_cJEPD2Hom?usp=drive_link)

---

## Pipeline overview

### Phase 1 — Biology → Data

| Step | What happens                                                             |
| ---- | ------------------------------------------------------------------------ |
| 1    | Fetch CDS + 300 bp upstream sequences from NCBI                          |
| 2a   | Compute GC content per gene                                              |
| 2b   | Compute 4-mer frequency fingerprint (256D vector)                        |
| 2c   | Score lux box match in upstream region                                   |
| 3    | PCA gene map from k-mer features                                         |
| 3b   | ESM2 protein embedding gene map (320D → 3D)                              |
| 3c   | ESMFold — predict 3D protein structure per gene                          |
| 4    | Create LuxZ: place on map · chimeric sequence · fold · nearest neighbour |
| 5    | Decode LuxZ back to biological features                                  |

### Phase 2 — Data → Images

| Step | What happens                                                        |
| ---- | ------------------------------------------------------------------- |
| 6    | Convert features to SD prompts (lux box → intensity, GC → colour)   |
| 7    | txt2img — one image per gene                                        |
| 7b   | Screen-blend ESMFold backbone onto SD image, unify with img2img     |
| 8    | img2img — apply biological style to TouchDesigner input             |
| 8b   | Blend TD input + protein structure composite                        |
| 8c   | ControlNet scribble — backbone geometry conditions SD generation    |
| 8d   | Lab colour blend: 8b colour × 8c-2 structure → final image per gene |

### Phase 3 — Images → Animation

| Step | What happens                                                                 |
| ---- | ---------------------------------------------------------------------------- |
| 8e   | Extract TD video frames, run img2img per gene                                |
| 9    | Screen-blend all gene videos frame-by-frame (biological weights)             |
| 9b   | Morph animation: 11 genes × 15 linear-blend frames, ordered by lux box score |
| 9c   | Dual-view GIF: organism video (left) vs gene morph (right)                   |
| 10   | Chained img2img: 16-frame bioluminescence activation loop                    |

---

## Key outputs

| File                                  | Description                                  |
| ------------------------------------- | -------------------------------------------- |
| `images/8d/{gene}_8d.png`             | Final composite image per gene               |
| `images/organism_morph.gif`           | Gene identity morphing sequence (150 frames) |
| `images/video_out/organism_video.gif` | Organism screen-blend video                  |
| `images/organism_dualview.gif`        | Side-by-side comparison                      |
| `images/lux_emission_luxZ_start.gif`  | Bioluminescence activation animation         |

---

## Utilities

**`fetch_lux_dna.py`** — downloads the _V. fischeri_ ES114 GenBank record from NCBI and extracts lux gene CDS sequences, 300 bp upstream promoter regions, and the operon block to `data/dna/`. Run this once before opening the notebook if the `data/dna/` directory is empty:

```bash
python3 fetch_lux_dna.py
```

**`extract_outputs.py`** — when the notebook is run on Google Colab, generated images are embedded in cell outputs as base64 data rather than saved to disk. Download the `.ipynb` from Colab, then run this script locally to extract every embedded image to `extracted_outputs/`:

```bash
python3 extract_outputs.py
```

---

## Technical documentation

See [TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md) for full step-by-step implementation details, parameter choices, and design rationale.
