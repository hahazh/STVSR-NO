# STNO (Arxiv 24)
This is the code of the paper "Space-Time Video Super-resolution with Neural Operator"

**The paper is currently undergoing peer review.**
# Abstract 

This paper addresses the task of space-time video super-resolution (ST-VSR). Existing methods generally suffer from inaccurate motion estimation and motion compensation (MEMC) problems for large motions. Inspired by recent progress in physics-informed neural networks, we model the challenges of MEMC in ST-VSR as a mapping between two continuous function spaces. Specifically, our approach transforms independent low-resolution representations in the coarse-grained continuous function space into refined representations with enriched spatiotemporal details in the fine-grained continuous function space. To achieve efficient and accurate MEMC, we design a Galerkin-type attention function to perform frame alignment and temporal interpolation. Due to the linear complexity of the Galerkin-type attention mechanism, our model avoids patch partitioning and offers global receptive fields, enabling precise estimation of large motions. The experimental results show that the proposed method surpasses state-of-the-art techniques in both fixed-size and continuous space-time video super-resolution tasks.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; justify-items: center; align-items: center;">
  <div style="text-align: center;">
    <img src="GIF/MoTIF_011.gif" alt="MoTIF GIF" width="640" height="360">
    <p>MoTIF Output</p>
  </div>
  <div style="text-align: center;">
    <img src="GIF/TMnet_011.gif" alt="TMnet GIF" width="640" height="360">
    <p>TMnet Output</p>
  </div>
  <div style="text-align: center;">
    <img src="GIF/zooming_slow_011.gif" alt="Zooming Slow GIF" width="640" height="360">
    <p>Zooming Slow Output</p>
  </div>
  <div style="text-align: center;">
    <img src="GIF/NOP_011.gif" alt="NOP GIF" width="640" height="360">
    <p>NOP Output</p>
  </div>
</div>
