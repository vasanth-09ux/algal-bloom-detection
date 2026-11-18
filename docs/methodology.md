# Technical Methodology: Exploratory Satellite Remote Sensing for Potential Algal Bloom Mapping

---

> [!IMPORTANT]
> **Exploratory Methodology Notice:**  
> This document details the mathematical, physical, and optical foundations of an **exploratory, unsupervised spectral mapping methodology**. The algorithms isolate surface water bodies and detect spectral anomalies whose reflectance characteristics align with floating algal biomass. This implementation does **NOT** constitute clinically or scientifically validated Harmful Algal Bloom (HAB) detection. Validation requires concurrent in-situ limnological/oceanographic vessel measurements (cell counts, taxonomic identification, and cyanotoxin ELISA assays).

---

## 1. Remote Sensing Foundations & Optical Principles

Harmful Algal Blooms (HABs) occur when aquatic micro-organisms multiply rapidly. Optical satellite sensors observe the combined reflectance of the water column, suspended matter, dissolved organic matter, and surface biomass:

1. **Chlorophyll-a Absorption & Green Scattering:** Photosynthetic pigments absorb solar radiation in the Blue ($\sim 440\text{--}490\text{ nm}$) and Red ($\sim 665\text{ nm}$) bands, creating a characteristic reflectance peak in the Green band ($\sim 560\text{ nm}$).
2. **Near-Infrared (NIR) Surface Scattering:** Water strongly absorbs radiation at wavelengths $>700\text{ nm}$ (NIR reflectance $<0.01\text{--}0.02$). However, dense surface scums and floating algal mats reflect strongly in NIR due to refractive index mismatches within cell structures.
3. **Short-Wave Infrared (SWIR) Baseline:** At $1610\text{ nm}$ (Sentinel-2 Band 11), both clean water and moderate algae exhibit near-total absorption, providing a stable baseline to remove atmospheric aerosol backscatter and ocean sun-glint.

---

## 2. Mathematical Formulations & Processing Pipeline

### 2.1 Surface Reflectance Conversion
Raw Sentinel-2 Level-2A data (`COPERNICUS/S2_SR_HARMONIZED`) represents surface reflectance scaled by 10,000:
$$\rho_{\text{band}} = \frac{\text{DN}}{10,000.0}$$

### 2.2 Water Body Delineation (NDWI)
To exclude terrestrial land and vegetation, open water is delineated using McFeeters' Normalized Difference Water Index:
$$\text{NDWI} = \frac{\rho_{\text{Green}} - \rho_{\text{NIR}}}{\rho_{\text{Green}} + \rho_{\text{NIR}}} = \frac{B3 - B8}{B3 + B8}$$

- Over clear water: $\rho_{\text{Green}} \gg \rho_{\text{NIR}} \implies \text{NDWI} > 0$.
- Over terrestrial vegetation: $\rho_{\text{NIR}} \gg \rho_{\text{Green}} \implies \text{NDWI} < 0$.
- In this pipeline, the empirical threshold is set to $\text{NDWI} > 0.20$.

### 2.3 Morphological Filtering
To eliminate surf noise, coastal breaking waves, and cloud-shadow boundaries, the initial binary mask is refined using mathematical morphology with a $3 \times 3$ square structuring element:
1. **Binary Opening:** $\mathbf{M} \circ \mathbf{S} = (\mathbf{M} \ominus \mathbf{S}) \oplus \mathbf{S}$ (erodes isolated noise pixels).
2. **Binary Closing:** $(\mathbf{M} \circ \mathbf{S}) \bullet \mathbf{S} = ((\mathbf{M} \circ \mathbf{S}) \oplus \mathbf{S}) \ominus \mathbf{S}$ (bridges narrow gaps and fills interior voids).

### 2.4 Normalized Difference Vegetation Index (NDVI)
Evaluated across isolated water pixels:
$$\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{Red}}}{\rho_{\text{NIR}} + \rho_{\text{Red}}} = \frac{B8 - B4}{B8 + B4}$$

### 2.5 Floating Algae Index (FAI)
Following Hu (2009), FAI measures the amplitude of the NIR reflectance peak above a baseline linearly interpolated between the Red and SWIR-1 bands:
- Nominal Sentinel-2 MSI wavelengths: $\lambda_{\text{Red}} = 665\text{ nm}$, $\lambda_{\text{NIR}} = 842\text{ nm}$, $\lambda_{\text{SWIR}} = 1610\text{ nm}$.
- Interpolation ratio:
  $$\text{Ratio} = \frac{\lambda_{\text{NIR}} - \lambda_{\text{Red}}}{\lambda_{\text{SWIR}} - \lambda_{\text{Red}}} = \frac{842 - 665}{1610 - 665} = \frac{177}{945} \approx 0.187302$$
- Baseline NIR calculation:
  $$\text{Baseline}_{\text{NIR}} = \rho_{\text{Red}} + (\rho_{\text{SWIR}} - \rho_{\text{Red}}) \times \text{Ratio}$$
- Floating Algae Index:
  $$\text{FAI} = \rho_{\text{NIR}} - \text{Baseline}_{\text{NIR}}$$

---

## 3. Unsupervised Clustering & Anomaly Mapping

1. **Multivariate Feature Assembly:** For every valid water pixel $p$, a 6D spectral feature vector is constructed:
   $$\mathbf{x}_p = \left[\rho_{\text{Blue}}, \rho_{\text{Green}}, \rho_{\text{Red}}, \rho_{\text{NIR}}, \text{NDVI}, \text{FAI}\right]^T \in \mathbb{R}^6$$

2. **K-Means Partitioning:**
   The water pixel population is clustered into $k=4$ groups using standard Lloyd's K-Means (`random_state=42`):
   $$\arg\min_{\mathcal{S}} \sum_{i=1}^{k} \sum_{\mathbf{x} \in \mathcal{S}_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$

3. **Bloom Likelihood Ranking Heuristic:**
   To identify which cluster represents potential algal bloom conditions, cluster centers $\boldsymbol{\mu}_i$ are scored using a weighted multi-attribute ranking:
   $$\text{Score}_i = 1.5 \cdot \boldsymbol{\mu}_{i, \text{NIR}} + 1.0 \cdot \left(\frac{\boldsymbol{\mu}_{i, \text{Green}}}{\boldsymbol{\mu}_{i, \text{Blue}} + 10^{-6}}\right) + 2.0 \cdot \boldsymbol{\mu}_{i, \text{FAI}}$$
   The cluster with the highest score is mapped to the **Potential Algal Bloom** class.

---

## 4. Empirical Cluster Metrics from Evaluated Scene

The table below reflects the exact cluster centers and scores produced by the pipeline over the coastal Sentinel-2 study scene:

| Cluster | Mean NIR Reflectance | Green / Blue Ratio | Mean FAI | Bloom Likelihood Score | Classification |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 0.0308 | 1.0930 | -0.0163 | 1.1066 | Clean Open Water |
| **1** | **0.0359** | **1.1720** | **-0.0278** | **1.1702** | **Potential Bloom Anomaly (Highest Score)** |
| **2** | 0.0917 | 1.0598 | -0.0225 | 1.1525 | Intermediate Turbidity |
| **3** | 0.0304 | 0.9763 | -0.0095 | 1.0030 | Deep Clear Water |

*Scene FAI Statistics on Water Pixels:*
- Minimum FAI: `-0.0598`
- Maximum FAI: `0.4522`
- Mean FAI: `0.0245`

---

## 5. Limitations & Scientific Boundaries

1. **Unsupervised Nature:** K-Means identifies natural mathematical groupings in spectral feature space. It cannot determine biological taxonomy (e.g. differentiating between *Microcystis*, *Noctiluca*, or harmless green algae).
2. **Submerged vs. Surface Scums:** FAI is sensitive to surface floating scums. Submerged algae deeper than $\sim 2\text{ meters}$ will have their NIR signal absorbed by water.
3. **Turbidity & Sediments:** Suspended minerals and estuarine sediment runoffs can elevate green and NIR reflectance, mimicking bloom signatures unless verified with localized optical measurements.
