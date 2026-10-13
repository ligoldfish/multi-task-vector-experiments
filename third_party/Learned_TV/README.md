

# Task Vectors, Learned Not Extracted: Performance Gains and Mechanistic Insights

<p align="center">
  <a href="https://hlyang2001.github.io/hly/">Haolin Yang</a>, et al.
  <br>
  <a href="https://arxiv.org/abs/2509.24169">arXiv</a> •
  <a href="https://openreview.net/forum?id=RGEbVZgf4E">OpenReview</a> 
  <br>
  <br>
  <a href="https://github.com/HLYang2001/Learned_TV/blob/master/LICENSE"><img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-yellow?style=flat&link=https%3A%2F%2Fgithub.com%2FHLYang2001%2FLearned_TV%2Fblob%2Fmaster%2FLICENSE"></a>
  <a href="https://openreview.net/forum?id=RGEbVZgf4E"><img src="https://img.shields.io/badge/ICLR_2026-Accepted-blue?link=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DxizpnYNvQq"></a>
  <a href="https://arxiv.org/abs/2509.24169"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2509.24169-red?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2410.04468"></a>
</p>

**This repo contains the official code for the following paper published at ICLR 2026:**

> Haolin Yang, et al. **"Task Vectors, Learned Not Extracted: Performance Gains and Mechanistic Insights"** *The Fourteenth International Conference on Learning Representations (ICLR)*, 2026.

Implemented by [Haolin Yang](https://hlyang2001.github.io/hly/), the primary contributor of the paper.

## Overview

### Abstract

*Large Language Models (LLMs) can perform new tasks from in-context demonstrations, a phenomenon known as in-context learning (ICL). Recent work suggests that these demonstrations are compressed into task vectors (TVs), compact task representations that LLMs exploit for predictions. However, prior studies typically extract TVs from model outputs or hidden states using cumbersome and opaque methods, and they rarely elucidate the mechanisms by which TVs influence computation. In this work, we address both limitations. First, we propose directly training Learned Task Vectors (LTVs), which surpass extracted TVs in accuracy and exhibit superior flexibility—acting effectively at arbitrary layers, positions, and even with ICL prompts. Second, through systematic analysis, we investigate the mechanistic role of TVs, showing that at the low level they steer predictions primarily through attention-head OV circuits, with a small subset of “key heads” most decisive. At a higher level, we find that despite Transformer nonlinearities, TV propagation is largely linear: early TVs are rotated toward task-relevant subspaces to improve logits of relevant labels, while later TVs are predominantly scaled in magnitude. Taken together, LTVs not only provide a practical approach for obtaining effective TVs but also offer a principled lens into the mechanistic foundations of ICL.*

### Summary figure

<p align="center">
<img src="graph_abs.jpg" width="70%" />
</p>

*(A) We directly train Learned Task Vectors (LTVs) to be injected, which influence model outputs through later layer updates. 

*(B) In the low-level interactions between TVs and later layers, the OV circuits of attention heads are the crucial components interacting with TVs to induce their effects. 

*(C) On a high level, subsequent layer updates act on TVs as a largely linear transformation of rotation and stretch, with the rotation aligning TVs with the relevant task subspace to promote prediction of task-related tokens.


## Set Up

### 0. Requirement

1. A GPU with more than 80GB VRAM and CUDA (Ver. `12.4` recommended) are strongly required to run all the experiments.
2. Network connection to `huggingface` is needed to download the pre-trained model. And a `huggingface` user token with access to the Llama- and Gemma-family models is recommended to run a part of the experiments.
3. `Anaconda` or `Miniconda` is needed.

### Clone the repository

```bash
git clone https://github.com/HLYang2001/Learned_TV.git
```

### Environment Installation

```bash
conda env create -f environment.yaml
conda activate ltv
```

### Folder Structure

data: contains files of datasets unavailable on HuggingFace
experiments: contains .py scripts to run the experiments
fig_data: contains the figures as the final outputs produced from the experiment results and the notebooks to create the figures
hs: contains the hidden state collections used in the experiments
logs: contains the experiment results in text format (e.g. tokens decoded from hidden states)
metric: contains the experiment results in tensor format 
score: contains experiment results related to prediction accuracies on different tasks
scripts: contains the shell scripts to call the .py experiment scripts.
utils: contains supporting functions to run the experiments


### Experiments Instruction

We provide the instructions to recreate the major figures in the paper below.

| Index | Which Figure | Shell Script | Output Path | Description |
| :---: | :---: | :---: | :---: | :---: |
| 1 | Figure 2 | `fig2.sh` | fig_data/layer_wise | Measure the final accuracy of injecting TV, FV, and LTV to the final token position of each layer |
| 2 | Figure 3 | `fig3.sh` | generation | Model generation results on the Myopic task under zero-shot, ICL, and TV/FV/LTV steering |
| 3 | Figure 4 | `fig4.sh` | fig_data/cossim | Intra- and inter-task cosine similarities of LTVs trained for different tasks |
| 4 | Figure 5 | `fig5_tab3.sh` | fig_data/compo_across | Performance of the task vector obtained by composing the LTVs trained on subordinate tasks on the composite task |
| 5 | Table 3 | `fig5_tab3.sh` | fig_data/compo_across | Performance of the LTV trained for one task on other tasks |
| 6 | Figure 6 | `fig6_7_40.sh` | fig_data/head_acc & head_abl | Accuracy achieved by reconstructing the TV effect through attention head OV circuits & Influence of ablating key heads leveraging TVs |
| 7 | Figure 40 | `fig6_7_40.sh` | fig_data/head_abl | Influence of whether to include the residual effect of TV on the reconstruction accuracy |
| 8 | Figure 7 | `fig6_7_40.sh` | fig_data/attn_dist & head_dist | Attention score distribution of the key heads leveraging TVs over token positions & and the distribution of key heads over layers |
| 9 | Figure 8 (A) | `fig8_9_10.sh` | fig_data/metrics| Layerwise values of the three metrics evaluated at the zero-shot, ICL, and TV-steered hidden states |
| 10 | Figure 8 (B) | `fig8_9_10.sh` | logs/{llm_name}/{dataset_name}/1/normal/constructed_tv.txt | Top tokens decoded from LTVs trained at early and late layers |
| 11 | Figure 9 (A) | `fig8_9_10.sh` | fig_data/recon_acc | Accuracy achieved by the linearly reconstructed task vector at each layer |
| 12 | Figure 9 (B) | `fig8_9_10.sh` | fig_data/hs_acc | Results of verifying whether the layer transformations act approximately linearly on hidden states |
| 13 | Figure 10 (A) & (B) | `fig8_9_10.sh` | fig_data/rot | Cosine similarity between the early layer LTV and label unembeddings before and after considering the rotation effect of the layer transformation & strength of the rotation |
| 14 | Table 1 | `tab1.sh` | logs/practical/{llm_name} | Leveraging the findings in the paper to improve classification performance |
| 15 | Table 2 | `tab2.sh` | fig_data/tab2.txt | Testing the performance of LTV compared to TV/FV in cases of multiple layer & token position injection and different prompt formats |

## Citation

If you find this work useful for your research, please cite [our paper](https://openreview.net/forum?id=RGEbVZgf4E):

```
@misc{yang2025localizingtaskrecognitiontask,
      title={Task Vectors, Learned Not Extracted: Performance Gains and Mechanistic Insights}, 
      author={Haolin Yang and Hakaze Cho and Kaize Ding and Naoya Inoue},
      year={2025},
      eprint={2509.24169},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.24169}, 
}
```