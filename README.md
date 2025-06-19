# LipNet 🔤👄

This is an implementation of the **LipNet** model, adapted to run on a smaller dataset. It is based on the research paper:  
**"LipNet: End-to-End Sentence-Level Lipreading"** by Yannis M. Assael et al.

## 📚 Paper
- Original Paper: [LipNet on arXiv](https://arxiv.org/abs/1611.01599)

## 🎯 Objective
To replicate a scaled-down version of the LipNet architecture to perform **lip reading** from video frames. This implementation focuses on sentence-level classification using a minimal dataset for quick prototyping and demonstration purposes.

## 📁 Project Structure
- `LipNet.ipynb`: The main Jupyter Notebook containing:
  - Data preprocessing
  - Model architecture
  - Training & validation loops
  - Evaluation metrics
  - Visualizations and prediction examples

## 🧠 Model Architecture
- **3D Convolutional Layers** for spatio-temporal feature extraction from video frames
- **Bidirectional GRUs** for sequential modeling
- **CTC Loss** (Connectionist Temporal Classification) for aligning predicted sequences with ground truth

## 🗃️ Dataset
- A small, custom dataset of videos with corresponding spoken sentences.
- Each sample consists of:
  - Preprocessed video frames (mouth region)
  - Corresponding text label

*Note:* Dataset is not publicly available in this repository due to size and licensing constraints. You'll need to use your own or publicly available alternatives such as GRID dataset.

