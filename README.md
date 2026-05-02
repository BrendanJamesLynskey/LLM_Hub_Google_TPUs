# Google TPUs

Deep technical tour of Google's Tensor Processing Unit programme &mdash; from the 2013 voice-search napkin maths through every generation (v1, v2, v3, v4, v4i, v5e, v5p, Trillium, Ironwood), the systolic-array heritage, the optical interconnect, and the XLA / JAX / Pallas software stack.

**Live index:** https://brendanjameslynskey.github.io/LLM_Hub_Google_TPUs/

## Presentations in this series

| # | Title | Status | Description |
|---|-------|--------|-------------|
| 01 | [History &amp; People &mdash; How the TPU Programme Began](https://brendanjameslynskey.github.io/Google_TPU_01_History_and_People/) | live | The 2013 voice-search projection, Norm Jouppi's career arc from Stanford MIPS to Google, David Patterson, the 22-day silicon-to-datacenter dash, AlphaGo, the ISCA 2017 / CACM 2020 / ISCA 2023 papers. |
| 02 | [Ten Years of TPUs &mdash; v1 to Ironwood](https://brendanjameslynskey.github.io/Google_TPU_02_Generations_Overview/) | live | Timeline of every TPU generation, what each one unlocked, the e-class / p-class fork, peak FLOPS / HBM / pod scaling, models trained on each. Interactive generation explorer. |
| 03 | [Systolic Arrays &mdash; The Matmul Engine Inside](https://brendanjameslynskey.github.io/Google_TPU_03_Systolic_Arrays/) | live | Kung &amp; Leiserson 1978, "Why Systolic Architectures?" 1982, Warp / iWarp, weight-stationary vs output-stationary vs row-stationary dataflows, how a 256&times;256 array maps a matmul. Interactive matmul wavefront. |
| 04 | [Inside TPU v1 &mdash; The 2015 Inference Chip](https://brendanjameslynskey.github.io/Google_TPU_04_Inside_TPU_v1/) | live | 28 nm, 256&times;256 INT8 systolic, 24 MiB unified buffer, 8 GiB DDR3, the brilliantly minimal CISC ISA, the 92 TOPS roofline and what Google learned about memory bandwidth. |
| 05 | [TPU v2 &amp; v3 &mdash; The Training Era Begins](https://brendanjameslynskey.github.io/Google_TPU_05_v2_v3_Training_Era/) | live | bfloat16 (Google Brain's invention), HBM arrives, two TensorCores per chip, the 2D torus pod, why v3 needed liquid cooling. |
| 06 | [TPU v4 &mdash; OCS, SparseCore &amp; Palomar](https://brendanjameslynskey.github.io/Google_TPU_06_v4_OCS_SparseCore/) | live | 7 nm, 3D torus, the Palomar 3D-MEMS optical circuit switch, SparseCore embedding accelerator, CMEM, the v4i inference sibling, PaLM at 6,144 chips. |
| 07 | [TPU v5e &amp; v5p &mdash; The Two-Track Fork](https://brendanjameslynskey.github.io/Google_TPU_07_v5e_v5p_Split/) | live | The efficiency / performance product fork, v5p as the Gemini training flagship, 95 GB HBM, the 8,960-chip pod, multislice over Jupiter DCN. |
| 08 | [Trillium &amp; Ironwood &mdash; v6e and v7](https://brendanjameslynskey.github.io/Google_TPU_08_Trillium_Ironwood/) | live | Trillium's 4.7&times; jump and 3rd-gen SparseCore, Ironwood's 4.6 PFLOPS FP8 / 192 GB HBM3e per chip, the 9,216-chip "age of inference" pod. |
| 09 | [Memory Hierarchy &amp; Numerics](https://brendanjameslynskey.github.io/Google_TPU_09_Memory_and_Numerics/) | live | VMEM, CMEM, HBM2 to HBM3e, the bf16 invention story, INT8 + FP8, accumulator widths, why each precision change doubled effective FLOPS. |
| 10 | [ICI, OCS &amp; the 3D Torus](https://brendanjameslynskey.github.io/Google_TPU_10_ICI_and_OCS/) | live | Custom-SerDes Inter-Chip Interconnect, 2D vs 3D torus, the Palomar 136-port optical switch, twisted-torus topologies, healing around faulty cubes, multipod over Jupiter DCN. |
| 11 | [The TPU Software Stack &mdash; XLA, JAX, Pallas](https://brendanjameslynskey.github.io/Google_TPU_11_Software_Stack/) | live | XLA / HLO / StableHLO, JAX transformations and sharding, GSPMD &amp; Shardy partitioning, Pallas / Mosaic kernels, PyTorch-XLA &amp; TorchTPU, MaxText, Pathways, Multislice. |
| 12 | [TPU vs GPU &mdash; Two Architectural Philosophies](https://brendanjameslynskey.github.io/Google_TPU_12_TPU_vs_GPU/) | live | Static-compiler vs dynamic-warp scheduling, scratchpad vs cache, AOT graph vs JIT kernel, 3D torus vs fat-tree InfiniBand, where each wins, the cloud-only constraint. |

## Where this fits

Part of the [LLMs hub](https://github.com/BrendanJamesLynskey/LLMs) &mdash; an index of presentation series for AI/LLM engineers. See the companion [NVIDIA GPU Architectures](https://github.com/BrendanJamesLynskey/LLM_Hub_NVIDIA_GPUs) and [CUDA Programming](https://github.com/BrendanJamesLynskey/LLM_Hub_CUDA) sub-hubs for the GPU side of the story.
