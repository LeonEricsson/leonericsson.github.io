---
layout: post
title: "DeepSeek OCR"
categories: []
year: 2025
type: paper
author: 
exturl: 
---
[WIP]

First a recap on the state of vision encoders. 

Vision Encoders are responsible for encoding the visual input to VLMs, often done with some ViT architecture. However, handling dynamic, or native input resolution is not supported in vanilla ViTs, so VLMs adopt technices to get around this. The core architectural challenge is adapting standard ViTs for high-resolution, variable-aspect-ratio inputs. Most ViTs are pre-trained with fixed-resolution inputs (e.g., 224x224, 384x384), which is a suboptimal convention. This fixed-input paradigm forces a trade-off: either downsample the image, which causes a significant loss of detailed information, or pad the image, which is computationally inefficient.

To process real-world data like documents, which are both high-resolution and have extreme aspect ratios, two primary strategies have emerged: tiling and adaptive resolution.
#### Tiling (InternVL, DeepSeek-VL2)
This is a "divide and conquer" or tile-based method. The high-resolution input is dynamically resized to match a pre-defined aspect ratio and then split into a grid of fixed-size tiles (e.g., $448 \times 448$ for InternVL 2.5, $384 \times 384$ for DeepSeek-VL2). A global "thumbnail" of the entire image, resized to a single tile, is also generated to retain global context. All tiles are then processed by a shared ViT encoder, and the resulting tokens from each tile are compressed (e.g., via "pixel unshuffle") and projected into the LLM's embedding space via an MLP adaptor. Fusion is not handled by the encoder; it's implicitly managed by the LLM's self-attention. The 2D layout is communicated by flattening the token sequence and inserting special structural tokens like `<tile_newline>`. The main critique is that this can lead to "excessive fragmentation" and a very long token sequence for the LLM.

#### Adaptive Resolution (Qwen2-VL)
This method adopts the NaViT paradigm, processing the full image directly via patch-based segmentation. The entire image is treated as a single canvas, and the ViT's patchification (e.g., $14 \times 14$ patches) is applied directly. This "Naive Dynamic Resolution" mechanism inherently produces a variable number of visual tokens that scales with the image's resolution. This strategy requires modifying the ViT; standard absolute position embeddings are removed and replaced with 2D Rotary Position Embedding (2D-RoPE), which can dynamically compute relative positional information. Consequently, fusion is handled _within the ViT itself_, as its global self-attention processes all patches simultaneously to create a holistic representation. The primary critique is that processing a single, massive image with global attention results in "massive activation memory consumption" due to the $N^2$ quadratic complexity of attention.

### DeepEncoder
DeepEncoder is architected to maintain low activation memory while producing a small, fixed number of highly informed vision tokens. It achieves this by explicitly separating the tasks of local perception and global knowledge fusion.

The architecture is a three-stage pipeline:

**Local Perception (SAM-base / ViTDet):** The high-resolution image (e.g., $1024 \times 1024$) is first processed by an 80M SAM-base (ViTDet) backbone. Using a fixed 16x16 patch size, this stage generates a very long sequence of patch tokens (e.g., 4096). The computational cost and activation memory are kept low because this component uses primarily window attention which confines self-attention to small, local regions.
    
**Local Downsampling (Conv 16x):** The resulting 4096 tokens are reshaped into a 2D grid and passed through a 2-layer convolutional module. This compressor performs a 16x downsampling, reducing the token count from 4096 to 256. Due to the small kernel size (3x3) and shallow depth, this CNN acts as a _local fuser_, merging features only from adjacent token positions.
    
**Global Fusion (CLIP-large):** The 256 locally-fused tokens are then fed into a 300M CLIP-large ViT, which acts as the "global attention" knowledge component. The initial patch embedding layer of this ViT is removed, as its input is already a sequence of tokens. This stack of Transformer blocks performs dense, global self-attention over the 256 tokens, finally enabling the long-range information fusion that was absent in the first two stages.
    
### DeepSeekMoE 3B A570M

The MoE fits into the VLM pipeline just as any other would, taking the compressed latent vision tokens from the DeepEncoder and mapping them to match the language models hidden dimension.

The MoE architecture is quite odd, at least now adays. It's got a very high activation ratio of 12.1% with **2** shared experts? It's also using MHA, not MLA or even GQA. The experts are also **very** wide for this size, with a intermediate size of 896, compared to the models hidden dimension of 1280 this means our granularity is as low as 2.85, far lower than anything we've seen recently from a MoE release. 

Quite odd MoE architecture.

