# Technical Implementation

This project translates real bacterial DNA into images and animations using Stable Diffusion. Every visual output — colour, brightness, structure — is determined directly by measurable properties of the DNA.

---

## Methodology

This project adopts a practice-based research methodology in which
the pipeline itself is the research instrument. Each technical decision
— the choice of ESM2 over k-mers, ControlNet over screen blend,
Lab blend over RGB compositing — was made iteratively through
making and testing, and is documented here alongside the rationale
for each choice.

The pipeline is structured in three phases:

- **Biological feature extraction** — making DNA readable as data
- **Image generation** — translating data into visual parameters
- **Animation** — assembling individual gene outputs into organism-level narratives

All code was developed in Python using a Jupyter notebook environment,
running on Google Colab (NVIDIA T4 GPU) and Apple Silicon (MPS backend).

---

## Dataset

### Source organism

**Species**: _Vibrio fischeri_ ES114  
**Genome**: Chromosome 2, NCBI accession `CP000021.2`  
_V. fischeri_ is a bioluminescent bacterium that colonises the light organ of the Hawaiian bobtail squid (_Euprymna scolopes_). Its lux operon is the canonical model system for quorum sensing — the mechanism by which bacteria coordinate gene expression in response to population density.

### Genes

The dataset covers the complete _lux_ operon plus two auxiliary genes: 10 natural genes and 1 synthetic gene created in this project.

| Locus tag | Gene     | Biological role                                                   |
| --------- | -------- | ----------------------------------------------------------------- |
| VF_A0921  | luxA     | Luciferase α-subunit — directly produces light                    |
| VF_A0920  | luxB     | Luciferase β-subunit — directly produces light                    |
| VF_A0923  | luxC     | Acyl reductase — builds the aldehyde fuel                         |
| VF_A0922  | luxD     | Acyl transferase — builds the aldehyde fuel                       |
| VF_A0919  | luxE     | Acyl-protein synthetase — builds the aldehyde fuel                |
| VF_A0918  | luxG     | Flavin reductase — provides FMNH₂ fuel to luciferase              |
| VF_A0924  | luxI     | Autoinducer synthase — generates the quorum sensing signal        |
| VF_A0925  | luxR     | Transcription factor — binds AHL and activates the lux operon     |
| VF_A0593  | luxT     | Regulator (indirect role)                                         |
| VF_A1058  | qsrP     | Quorum sensing related (indirect role)                            |
| —         | **luxZ** | **Synthetic hybrid gene** — designed in this project (see Step 4) |

### Sequence data fetched per gene

- **CDS** (`lux_cds.fasta`) — the coding sequence (the gene itself, in ATCG)
- **Upstream 300 bp** (`lux_upstream_300bp.fasta`) — the regulatory region immediately before the gene, where the lux box lives

Both are downloaded automatically from NCBI via BioPython (`Entrez.efetch`) when Step 1 is run.

---

## Pipeline Overview

### Phase 1 — Biology → Data

| Step    | Input                   | Output                                                 |
| ------- | ----------------------- | ------------------------------------------------------ |
| 1       | NCBI accession          | Raw DNA sequences                                      |
| 2       | DNA                     | GC content · k-mer fingerprint · **lux box score**     |
| 3 / 3b  | DNA features + proteins | Gene map (PCA · ESM2)                                  |
| 3c      | Protein sequences       | **3D backbone structures (ESMFold)**                   |
| **4 ★** | **Gene map**            | **LuxZ — synthetic gene: placed · sequenced · folded** |
| 5       | LuxZ coordinate         | Decoded biological features                            |

### Phase 2 — Data → Images

| Step | Input                 | Output                               |
| ---- | --------------------- | ------------------------------------ |
| 6    | Biological features   | SD prompts per gene                  |
| 7    | Prompts               | Generated images (txt2img)           |
| 7b   | SD images + backbones | Protein structure composite          |
| 8    | TD input + prompts    | Biologically styled TD image         |
| 8b   | TD + 7b composite     | Blended composite                    |
| 8c   | TD img2img + backbone | ControlNet-shaped image              |
| 8d   | 8b + 8c-2             | **Final image per gene** (Lab blend) |

### Phase 3 — Images → Animation

| Step | Input              | Output                              |
| ---- | ------------------ | ----------------------------------- |
| 8e   | TD video + prompts | Per-gene styled video frames        |
| 9b   | 8d images          | Gene identity morph (lux box order) |
| 9c   | 8e frames          | Organism video (screen blend)       |
| 9d   | 9b + 9c            | Dual-view side-by-side              |
| 10   | TD image           | Bioluminescence activation loop     |
| 10b  | luxI 8d image      | Bioluminescence activation loop     |

---

## ★ LuxZ — The Gene That Doesn't Exist

> **LuxZ** was invented for this project, but it is grounded at every level in real biology.
>
> **Ａ. Placed at a biologically meaningful coordinate.** Placed 70% toward the light-emission cluster (luxA/B) and 30% toward the quorum-sensing cluster (luxI/R) on the ESM2 gene map — encoding the hypothesis of a hybrid regulator-emitter.
>
> **B. Given a real chimeric amino acid sequence.** luxA and luxI were aligned with BioPython. At each aligned position, a residue was sampled with p=0.70 from luxA and p=0.30 from luxI (seed=42). The result is a protein whose residues are genuinely drawn from **both parent genes** in proportion to their biological weighting — not an interpolation.
>
> **C. Folded into a genuine predicted 3D structure.** The chimeric sequence was run through ESMFold — the same structure-prediction pipeline used for all natural genes — producing a real backbone trace.
>
> **D. Full participant in every downstream step.** LuxZ enters Step 6 with decoded biological features (lux box score 0.62, its own GC content and k-mer profile) and is treated identically to the 10 natural genes from that point on.

---

## Environment and Dependencies

```bash
pip install biopython scikit-learn scikit-image matplotlib numpy diffusers transformers accelerate einops sentencepiece
```

The pipeline supports three compute backends, detected automatically:

| Backend | Hardware              | dtype     |
| ------- | --------------------- | --------- |
| `cuda`  | NVIDIA GPU            | `float16` |
| `mps`   | Apple Silicon (M1–M4) | `float32` |
| `cpu`   | Any machine           | `float32` |

SD model (~4 GB) and ESMFold (~2.7 GB) download and cache automatically on first run.

---

## Pipeline

### Step 1 — Fetch DNA Sequences from NCBI

Downloads CDS and 300 bp upstream sequences for all 10 lux genes from chromosome 2 of _V. fischeri_ ES114 (`CP000021.2`) using `Bio.Entrez`. Sequences are saved to `data/dna/` and cached; the download is skipped on subsequent runs if the files exist.

**Outputs**: `data/dna/lux_cds.fasta`, `data/dna/lux_upstream_300bp.fasta`

---

### Step 2 — Biological Feature Extraction

Three quantitative features are computed per gene. These features directly **control the visual output** in later steps.

#### Step 2a — GC Content

The fraction of G/C bases in the CDS. G–C pairs form three hydrogen bonds (vs two for A–T), making high-GC sequences more thermally stable and expressed under tightly controlled conditions. GC content is mapped to colour temperature in the image prompt:

| GC range  | Colour mapped                   |
| --------- | ------------------------------- |
| > 0.36    | Cold electric blue-white        |
| 0.33–0.36 | Blue-green bioluminescent       |
| < 0.33    | Warm amber-green bioluminescent |

![GC content bar chart](images/gc_content.png)

_GC content for each gene. Higher GC = more thermally stable expression under tightly controlled conditions. This value sets the colour temperature of every generated image._

#### Step 2b — 4-mer Frequency Fingerprint

Every possible 4-base substring (k=4, 4⁴ = 256 possible k-mers: AAAA…TTTT) is counted across the CDS and normalised by length. The result is a 256-dimensional vector that encodes the gene's sequence identity in a length-independent, machine-readable form.

#### Step 2c — Lux Box Score

The lux box is a 20-base consensus sequence (`ACCTGTAGGATCGTACAGGT`)（ref: https://pubmed.ncbi.nlm.nih.gov/2801220/) in the upstream regulatory region. When AHL (the quorum sensing signal) reaches threshold, LuxR binds this site and activates the entire operon. The score is computed by sliding a 20-base window across the upstream sequence and recording the best fractional match to the consensus (1.0 = perfect match).

This score is the single most important feature: it directly **controls light intensity in every generated image**.

```
Lux box score → SD light intensity mapping
──────────────────────────────────────────────────────
luxI  ██████████████████░░  0.90
luxR  ████████████████░░░░  0.80
qsrP  █████████████░░░░░░░  0.65
luxZ  ████████████░░░░░░░░  0.62  → decoded
luxE  ████████████░░░░░░░░  0.60
luxC  ████████████░░░░░░░░  0.60
luxT  ███████████░░░░░░░░░  0.55
luxG  ███████████░░░░░░░░░  0.55
luxA  ██████████░░░░░░░░░░  0.50
luxB  ██████████░░░░░░░░░░  0.50
luxD  ██████████░░░░░░░░░░  0.50
──────────────────────────────────────────────────────
Each ░ = 0.05 units. Filled bar = score × 20 segments.
```

**Cached outputs**: `data/gc_data.json`, `data/luxbox_data.json`

---

### Step 3 — Gene Map with PCA

#### Step 3 — PCA on hand-crafted features

Each gene is described by a 257-dimensional vector (1 GC value + 256 k-mer frequencies). PCA reduces this to 3 dimensions, placing each gene as a point in a navigable latent space. The axes of this space correspond to directions of maximum variance across the gene set.

**Key property**: Because the input features are interpretable, this space is approximately invertible — any 3D coordinate can be projected back into an approximate GC content and k-mer profile.

#### Step 3b — ESM2 Protein Embeddings

The protein sequence translated from each CDS is embedded using **ESM2** (`facebook/esm2_t6_8M_UR50D`), a protein language model trained on 250 million sequences. The 320-dimensional ESM2 embeddings capture evolutionary and functional relationships that raw k-mer frequencies cannot. PCA is applied to project these to 3D.

**Key distinction from Step 3**: ESM2 embeddings are not invertible — the axes carry no human-readable labels. Proximity in this space means functional kinship, but the coordinates cannot be decoded back to sequence properties.

**Why ESM2 over k-mers**:

|                    | k-mer PCA                                 | ESM2                                           |
| ------------------ | ----------------------------------------- | ---------------------------------------------- |
| What it encodes    | Raw sequence statistics (letter patterns) | Evolutionary and functional relationships      |
| Trained on         | —                                         | 250M protein sequences                         |
| Interpretable axes | ✅ Can decode back to GC / k-mer profile  | ❌ Axes carry no readable labels               |
| Functional kinship | ❌ Similar k-mers ≠ similar function      | ✅ Functionally related genes cluster together |
| Used for           | Placing LuxZ on the gene map (Step 4)     | Finding LuxZ's nearest neighbours (Step 4d)    |

![k-mer PCA vs ESM2 comparison](images/gene_map_comparison.png)

_Comparison of the k-mer PCA map (left) and the ESM2 protein embedding map (right). In the k-mer map, genes cluster by raw sequence statistics. In the ESM2 map they cluster by functional role — genes doing **similar biological jobs** end up near each other even when their DNA sequences have diverged over evolution._

#### Step 3c — ESMFold Protein Structure Prediction

ESMFold predicts the 3D folded structure of each lux protein from its amino acid sequence alone, without experimental data. Per gene, the pipeline produces:

- A `.pdb` file containing the full atomic coordinates
- A `_structure.png` backbone trace (Cα atoms, coloured N→C terminus) saved to `images/structures/`

These backbone images are used as ControlNet conditioning inputs in Step 8c-2.

![All protein backbone traces](images/structures/all_structures.png)

_ESMFold-predicted backbone traces for all 11 lux proteins. Each line traces the Cα backbone from N-terminus (blue) to C-terminus (red). No experimental data was used — these are pure predictions from amino acid sequence. These shapes become the ControlNet conditioning signal in Step 8c-2, actively guiding how SD redraws each gene image._

---

### Step 4 — LuxZ: Synthetic Gene Creation

#### Step 4 — Coordinate placement

LuxZ is a hypothetical gene placed on the PCA gene map at a position 70% toward the light-production cluster (luxA/B) and 30% toward the quorum sensing cluster (luxI/R). This is a deliberate biological hypothesis: a gene that both senses population density and contributes to light emission.

![Gene map with LuxZ placed](images/gene_map_luxZ.png)

_The PCA gene map with LuxZ placed at 70% toward the luxA/B cluster (light emitters) and 30% toward luxI/R (quorum sensing). LuxZ doesn't exist in nature — its position encodes the biological hypothesis that it would be a hybrid gene capable of both sensing the environment and contributing to light production._

#### Step 4b — Chimeric protein sequence

LuxZ is **given a real amino acid sequence** by aligning luxA and luxI with BioPython and sampling each position stochastically (p=0.70 luxA, p=0.30 luxI, seed=42). The result is a chimeric protein whose residues are drawn from both parents in proportion to their biological weighting.

#### Step 4c — ESMFold structure

The chimeric luxZ sequence is folded with ESMFold, producing a genuine predicted 3D structure. This replaces the interpolated backbone used in earlier steps.

#### Step 4d — Nearest biological neighbour

The chimeric luxZ is embedded with ESM2 and cosine similarity is computed against all 10 natural lux genes and a curated set of water/environment sensing proteins. This grounds the artistic reframing of luxZ as a water-sensing hybrid gene.

![LuxZ nearest neighbours](images/luxZ_nearest_neighbours.png)

_LuxZ's nearest biological neighbours ranked by ESM2 cosine similarity. The chimeric protein sits closest to luciferase subunits and environment-sensing proteins — a result driven by its actual amino acid sequence (70% luxA, 30% luxI), not by the coordinate placement in Step 4._

---

### Step 5 — Decode LuxZ Properties

LuxZ's 3D PCA coordinate is reverse-projected into the original 257-dimensional feature space to recover approximate GC content and k-mer profile. The lux box score is estimated as a weighted average of its parent genes (luxA and luxI, weighted 70/30). These decoded properties are fed into Step 6 on equal footing with the natural genes.

---

### Step 6 — Biological Features → Stable Diffusion Prompts

Each gene's three features are mapped to prompt elements using deterministic rules:

| Feature             | → Prompt element                                                                  |
| ------------------- | --------------------------------------------------------------------------------- |
| Lux box score (0–1) | Light intensity: dormant / faint / softly glowing / radiant / blazing             |
| GC content          | Colour temperature: cold blue-white / blue-green / warm amber-green               |
| Biological role     | Scene role: produces light / builds substrate / quorum sensing signal / regulator |

Every prompt follows the template:

```
"microscopic view of a single bacterial cell named {gene}, {colour} bioluminescent light,
{role}, dark deep ocean background, scientific macro photography,
bioluminescent organism, photorealistic, cinematic lighting, 8k"
```

Prompts are saved to `data/sd_prompts.json` and reused by all downstream steps.

---

### Step 7 — Stable Diffusion Image Generation

**Model**: `runwayml/stable-diffusion-v1-5` (text-to-image)  
**Parameters**: seed=42 per gene, 30 inference steps, 512×512

One image is generated per gene using its Step 6 prompt. The images are the first fully visual representation of the biological data — colour and luminosity differ between genes because the prompts differ, and the prompts differ because the DNA differs.

**Output**: `images/sd_{gene}.png`

![All 11 SD-generated gene images](images/sd_all_genes.png)

_All 11 gene images generated purely from text prompts derived from DNA analysis.None of these differences were chosen by hand; they all follow deterministically from the DNA._

#### Step 7b — Protein Structure Composite

Each gene's ESMFold backbone trace (Step 3c) is screen-blended onto its Step 7 SD image using **NumPy array operations**. The composite is passed through img2img (strength=0.4) to unify the blended result into a coherent scene. The protein structure appears as a glowing geometry inside the bioluminescent cell.

**Output**: `images/composite/{gene}_refined.png`

![All Step 7b composites](images/composite/all_refined_s6.png)

_Step 7b: ESMFold backbone traces screen-blended onto each SD image, then unified with a second img2img pass. The glowing geometric structures visible inside each cell are the actual predicted protein shapes for that gene._

---

### Step 8 — img2img: Apply Biological Style to TouchDesigner Input

**Model**: `StableDiffusionImg2ImgPipeline` (runwayml/stable-diffusion-v1-5)

Rather than generating from noise, the pipeline initialises from a TouchDesigner-generated visual (`TDimage1.tif`) and transforms it using each gene's prompt. The `strength` parameter (0.6) controls how far SD departs from the input: lower preserves the TD composition, higher gives SD creative freedom.

**Output**: `images/img2img_{gene}.png`

![All Step 8 img2img outputs](images/img2img_all_genes.png)

_Step 8: the TouchDesigner input image transformed with each gene's biological prompt via img2img. The TD composition (cell cluster layout) is the starting structure; SD overlays the bioluminescence biology on top. Colour and glow intensity still come entirely from the DNA data._

#### Step 8b — TD × Protein Structure Composite

Three sources are blended per gene:

1. TD input image (50% linear blend weight)
2. Step 7b refined composite (50% linear blend weight)
3. Screen-blend of TD on top to preserve bright structural elements

The result is passed through img2img at strength=0.6.

**Output**: `images/td_composite/{gene}_td_composite.png`

![All Step 8b TD composites](extracted_outputs/cell_043_output_26.png)

_Step 8b: the TD cell cluster layout (50%), the protein-embedded SD image from Step 7b (50%), and a screen blend layer, all unified with img2img. The TouchDesigner composition provides spatial structure; the protein geometry and bioluminescent colour are layered in._

#### Step 8c — Screen Blend Attempt (superseded by Step 8c-2)

The ESMFold backbone trace is screen-blended onto the Step 8 img2img output, then the blended result is passed through img2img at strength=0.45 to unify the layers.

**Why it didn't work**: The backbone was visible but structurally disconnected from the rest of the image. This limitation led directly to the ControlNet approach in Step 8c-2, where backbone geometry is injected into the generation process itself rather than pre-blended into the input.

**Output**: `images/8c/{gene}_8c.png` (not used in downstream steps)

#### Step 8c-2 — ControlNet Backbone + TD img2img

**Model**: `StableDiffusionControlNetImg2ImgPipeline` + `lllyasviel/control_v11p_sd15_scribble`

Step 8c-2 feeds the backbone geometry directly into the SD generation process via **ControlNet scribble conditioning**, so protein structure actively reshapes the image rather than sitting on top of it. The backbone trace is inverted before conditioning (ControlNet scribble expects white background, dark lines).

**Why ControlNet over simple screen blend**: Step 8c placed the backbone on top of the image as a screen-blended overlay, then ran img2img to unify it. The structure was visible but cosmetic — SD smoothed the blend without reorganising the composition around the protein geometry. ControlNet injects the backbone as a conditioning signal into the denoising process itself, so the protein shape actively guides how every pixel is reconstructed rather than being received as a pre-composed input. `controlnet_conditioning_scale=2.5` controls how strongly the backbone overrides the img2img base.

**Parameters**: strength=0.65, guidance_scale=3.0, controlnet_conditioning_scale=2.5

**Output**: `images/8c2/{gene}_8c2.png`

_Step 8c-2 output for luxZ. ControlNet scribble conditioning fed the inverted backbone geometry directly into the SD generation process._

#### Step 8d — Lab Blend: 8b Colour × 8c-2 Structure

A pure mathematical blend with no SD involved. Both Step 8b and Step 8c-2 outputs are converted to CIE Lab colour space:

- **L channel** (lightness/depth): 50% from 8b + 50% from 8c-2
- **ab channels** (colour): fully from 8b

**Why Lab blend**: The bioluminescent colour palette is preserved intact; the protein structure is incorporated only into the luminance layer. Fully deterministic and reproducible.

**Output**: `images/8d/{gene}_8d.png`

## ![Step 8d](images/8d/all_8d.png)

### Step 8e — Video Input × Lux Biology

**Model**: `StableDiffusionImg2ImgPipeline` (runwayml/stable-diffusion-v1-5)

Only five genes are processed — the ones with the most visually distinct biological signals: luxA, luxB (direct light emitters), luxI, luxR (quorum sensing regulators), and luxZ (synthetic hybrid).

**Input video**: `TDMovieOut.0.mov` — 1280×720, 60 fps, 12.4 s  
**Extraction**: 5 fps → ~62 frames, rescaled to 512×512  
**Parameters**: strength=0.6, 20 inference steps, guidance_scale=7.5, seed=42 per gene  
**Output**: `images/video_out/{gene}_out.gif` (5 fps loop)

![LuxZ video output](images/video_out/luxZ_out.gif)

_Step 8e output for luxZ. Each frame of the TouchDesigner video was independently transformed with luxZ's biological prompt (softly glowing, quorum-sensing hybrid)._

---

### Step 9 — Organism Assembly and Morphing

#### Step 9 — Organism Video: Screen-Blend All Gene Videos

Per-gene video frames from Step 8e are screen-blended together frame-by-frame using biological weights. Screen blending is the physically correct compositing mode for light sources: `result = 1 − (1−a)(1−b)`. Weights reflect direct vs indirect roles in bioluminescence.

| Gene       | Weight | Rationale                          |
| ---------- | ------ | ---------------------------------- |
| luxA, luxB | 1.00   | Direct light emitters (luciferase) |
| luxZ       | 0.85   | Synthetic hybrid                   |
| luxI, luxR | 0.50   | Quorum sensing regulators          |

**Output**: `images/video_out/organism_video.gif`

![Organism video](images/video_out/organism_video.gif)

_Organism video: all five gene video sequences screen-blended frame-by-frame._

#### Step 9b — Gene Identity Morph Animation

All 11 Step 8d gene images are morphed in sequence ordered by lux box score (luxI → luxR → qsrP → luxZ → luxE → luxC → luxT → luxG → luxB → luxA → luxD). Between each adjacent pair, 15 linear-blend frames are computed:

```python
frame = (1 - t) * img_a + t * img_b   # t ∈ [0, 1)
```

No SD or screen blending — pure pixel interpolation preserving 8d image detail.

**Output**: `images/organism_morph.gif` (150 frames at 12 fps, ~12.6 s)

![Step 9b colab float16 outputs](extracted_outputs/cell_055_output_1.gif)

_Generated on Colab T4 (float16)._

#### Step 9c — Comparision: Organism vs Gene Identity

**Output**: `images/organism_dualview.gif` at 12 fps

![Step 9c colab float16 outputs](extracted_outputs/cell_057_output_1.gif)

---

### Step 10 — Animation: luxI Emits the Signal

**Model**: `StableDiffusionImg2ImgPipeline` (runwayml/stable-diffusion-v1-5)  
**Starting image**: `TDimage1.tif`

While Steps 9b and 9c assemble the organism from existing 8d images, Step 10 takes a **different approach** entirely: it generates new frames from scratch using chained img2img, letting SD narrate the biological story rather than compositing pre-existing outputs.

Picks up at luxI — the gene with the highest quorum sensing score — and traces the full activation arc frame by frame:

```
dark cell → luxI fires → AHL builds → lux box ON → luxA/B erupt → sustained glow
```

Each frame is passed as the input image for the next, so the visual state evolves continuously. A bloom effect is applied post-generation to amplify the bioluminescent glow.

16 frames are defined covering the full bioluminescence activation arc:

| Frame range | Quorum state (QS) | Biological event                                            |
| ----------- | ----------------- | ----------------------------------------------------------- |
| 0–2         | 0.00–0.10         | Cell dark — luxI quiet, no autoinducer                      |
| 3–5         | 0.20–0.40         | luxI producing AHL, signal accumulating                     |
| 6           | 0.50              | lux box activated — LuxR binds, operon switches ON          |
| 7–9         | 0.60–0.80         | luxC/D/E produce aldehyde fuel; luxA/B luciferase activates |
| 10–12       | 0.90–1.00         | Maximum bioluminescence, LuxZ contributing                  |
| 13–15       | 0.95–0.70         | Sustained emission, slight decay toward loop                |

**Parameters**: 35 inference steps, strength per frame (0.40–0.55), seed=42+i per frame  
**Output**: `images/lux_emission_glow.gif` (32 frames at 120 ms/frame, ~3.8 s loop)

![luxI bioluminescence activation loop](images/lux_emission_glow.gif)

_Step 10 output with bloom effect generated from TD image input_

---

### Step 10b — Bioluminescence Animation from luxI 8d Starting Frame

**Model**: `StableDiffusionImg2ImgPipeline` (runwayml/stable-diffusion-v1-5)  
**Starting image**: `images/8d/luxI_8d.png`

Same as Step 10, but with a different starting image. Where Step 10 begins from the organism composite (a TD-derived frame), Step 10b begins from **luxI's Step 8d output** — the Lab blend image that has the ESMFold protein backbone baked into its luminance layer.

**Parameters**: 35 inference steps, strength per frame (0.40–0.55), seed=42+i per frame  
**Output**: `images/lux_emission_luxZ_start.gif` (32 frames at 120 ms/frame, ~3.8 s loop)

![Step 10b — luxI 8d activation loop](images/lux_emission_luxZ_start.gif)

_Step 10b output starting from luxI's 8d image. The opening frame carries the protein backbone geometry from Step 8d; the chained img2img then drives the cell from dormant through full bioluminescence activation._
