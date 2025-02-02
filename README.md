# STNO (Arxiv 24)
This is the code of the paper "Space-Time Video Super-resolution with Neural Operator"

**The paper is currently undergoing peer review. detail information will be coming soon.**
# Abstract 

This paper addresses the task of space-time video super-resolution (ST-VSR). Existing methods generally suffer from inaccurate motion estimation and motion compensation (MEMC) problems for large motions. Inspired by recent progress in physics-informed neural networks, we model the challenges of MEMC in ST-VSR as a mapping between two continuous function spaces. Specifically, our approach transforms independent low-resolution representations in the coarse-grained continuous function space into refined representations with enriched spatiotemporal details in the fine-grained continuous function space. To achieve efficient and accurate MEMC, we design a Galerkin-type attention function to perform frame alignment and temporal interpolation. Due to the linear complexity of the Galerkin-type attention mechanism, our model avoids patch partitioning and offers global receptive fields, enabling precise estimation of large motions. The experimental results show that the proposed method surpasses state-of-the-art techniques in both fixed-size and continuous space-time video super-resolution tasks.

# test code

### test fix-scale space-time video super-resolution
```
cd src/test_script

python test_vid4.py --datapath REDSPATH --outputpath  OUTPUTPATH --weight PATHTOWEIGHT
```

### test continuous space-time video super-resolution
```
cd src/test_script

python test_contin.py --datapath REDSPATH --outputpath  OUTPUTPATH --weight PATHTOWEIGHT
```

### pretrained weight
[pretrained model]( https://pan.baidu.com/s/1PA7IoclyZsDXA7EhNlGQjA?pwd=8n5e)
password: 8n5e 

## visual comparisons
<table>
  <tr>
    <td align="center">
      <img src="GIF/MoTIF_011.gif" alt="MoTIF GIF" width="640" height="360"><br>
      MoTIF Output
    </td>
    <td align="center">
      <img src="GIF/TMnet_011.gif" alt="TMnet GIF" width="640" height="360"><br>
      TMnet Output
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="GIF/zooming_slow_011.gif" alt="Zooming Slow GIF" width="640" height="360"><br>
      Zooming Slow Output
    </td>
    <td align="center">
      <img src="GIF/NOP_011.gif" alt="NOP GIF" width="640" height="360"><br>
      NOP Output (Ours)
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img src="GIF/MoTIF_015.gif" alt="MoTIF GIF" width="640" height="360"><br>
      MoTIF Output
    </td>
    <td align="center">
      <img src="GIF/TMnet_015.gif" alt="TMnet GIF" width="640" height="360"><br>
      TMnet Output
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="GIF/zooming_slow_015.gif" alt="Zooming Slow GIF" width="640" height="360"><br>
      Zooming Slow Output
    </td>
    <td align="center">
      <img src="GIF/NOP_015.gif" alt="NOP GIF" width="640" height="360"><br>
      NOP Output (Ours)
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img src="GIF/MoTIF_000_test.gif" alt="MoTIF GIF" width="640" height="360"><br>
      MoTIF Output
    </td>
    <td align="center">
      <img src="GIF/TMnet_000_test.gif" alt="TMnet GIF" width="640" height="360"><br>
      TMnet Output
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="GIF/zooming_slow_000_test.gif" alt="Zooming Slow GIF" width="640" height="360"><br>
      Zooming Slow Output
    </td>
    <td align="center">
      <img src="GIF/NOP_000_test.gif" alt="NOP GIF" width="640" height="360"><br>
      NOP Output (Ours)
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img src="GIF/MoTIF_001_test.gif" alt="MoTIF GIF" width="640" height="360"><br>
      MoTIF Output
    </td>
    <td align="center">
      <img src="GIF/TMnet_001_test.gif" alt="TMnet GIF" width="640" height="360"><br>
      TMnet Output
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="GIF/zooming_slow_001_test.gif" alt="Zooming Slow GIF" width="640" height="360"><br>
      Zooming Slow Output
    </td>
    <td align="center">
      <img src="GIF/NOP_001_test.gif" alt="NOP GIF" width="640" height="360"><br>
      NOP Output (Ours)
    </td>
  </tr>
</table>

# Acknowledgment
Our code is built on

 [Zooming-Slow-Mo-CVPR-2020](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)

 [open-mmlab](https://github.com/open-mmlab)

 [bicubic_pytorch](https://github.com/sanghyun-son/bicubic_pytorch)

 [IFRNet](https://github.com/ltkong218/IFRNet)

 [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI)
 
 [Galerkin Transformer](https://github.com/scaomath/galerkin-transformer)
 We thank the authors for sharing their codes!