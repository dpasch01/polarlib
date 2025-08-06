# Polarlib / PARALLAX

A framework that integrates polarization knowledge into machine learning tasks, enhancing classification accuracy in applications like misinformation detection and hate-speech identification.

### PARALLAX Quickstart

PARALLAX enables the use of polarization knowledge in supervised machine learning tasks, enhancing classifiers for misinformation detection. It builds on the PKG to represent article-level polarization knowledge through micro-PKGs and applies a GNN (Graph Neural Network) to incorporate polarization as a feature for existing classifiers.

![PARALLAX Framework Architecture](parallax.png)

*Figure 4: PARALLAX framework architecture for integrating polarization knowledge into misinformation detection.*

1. **Supplementary Corpus Curation:** Gathers a Supplementary Corpus of domain-relevant articles that provide contextual polarization knowledge.
2. **PKG Construction:** Constructs a Polarization Knowledge Graph for each domain, representing entity interactions and affiliations.
3. **Micro-PKG Construction:** Builds micro-PKGs for individual articles in the dataset, with sentiment-driven entity relations.
4. **Embedding Generation:** Learns polarization embeddings for micro-PKGs.
5. **Polarization-Aware Training:** Incorporates FlexKGNN, a GNN model designed to combine micro-PKG features with traditional classifier features, achieving enhanced classification performance.
