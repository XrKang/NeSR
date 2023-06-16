# Continuous Spectral Reconstruction from RGB Images via Implicit Neural Representation

> [Continuous Spectral Reconstruction from RGB Images via Implicit Neural Representation](https://link.springer.com/chapter/10.1007/978-3-031-25072-9_6), In *ECCVW* 2022. <br />
> Ruikang Xu, Mingde Yao, Chang Chen, Lizhi Wang, and Zhiwei Xiong. <br /> 
> MIPI Workshop Best Paper Honorable Mention. <br />

****

## Dependencies
* Python 3.7.0, PyTorch 1.13.0.
* NumPy 1.21.2, OpenCV 4.5.3, Pillow, Imageio, SciPy. 
  
****

## Quick Start
We take spectral reconstruction with an arbitrary number of spectral bands on the ICVL dataset as example. 

* Prepare the ICVL dataset before inference, it can be downloaded from this [link](https://icvl.cs.bgu.ac.il/hyperspectral/). 

* Download the pre-trained model from [BaiduYun](https://pan.baidu.com/s/10ZHsc7-2S5-NzC_BY9HPow?pwd=eccv) (Access code: eccv).  


* Inference with pre-trained model:
  ```
  cd ./code && python test.py
  ```

****

## Contact
Any question regarding this work can be addressed to xurk@mail.ustc.edu.cn and mdyao@mail.ustc.edu.cn.

****


## Citation
If you find our work helpful, please cite the following paper.
```
@inproceedings{xu2022continuous,
  title={Continuous spectral reconstruction from rgb images via implicit neural representation},
  author={Xu, Ruikang and Yao, Mingde and Chen, Chang and Wang, Lizhi and Xiong, Zhiwei},
  booktitle={European Conference on Computer Vision Workshop (ECCVW)} ,
  year={2022}
}
```
