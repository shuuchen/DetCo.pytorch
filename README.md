# DetCo.pytorch
An unofficial PyTorch implementation of DetCo [paper](https://arxiv.org/pdf/2102.04803.pdf). Official implementation can be found [here](https://github.com/xieenze/DetCo).

<img width="746" alt="スクリーンショット 2021-03-30 11 39 04" src="https://user-images.githubusercontent.com/13246825/112925323-a5cab780-914c-11eb-80f9-19199fb439d6.png">

This implementation is based on the official implementation of [MoCo](https://github.com/facebookresearch/moco) and [PyTorch resnet code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py). Though list is mentioned in original paper, this implementation use pure tensors without list, which enables parallel computing.

## Requirements
```
pip install -r requirements.txt
```

## Usage
- An example as in [main_decto.py](https://github.com/shuuchen/DetCo.pytorch/blob/main/main_decto.py):
```python
...
model = DetCo(resnet50(), args.detco_dim, args.detco_k, args.detco_m, args.detco_t)
...
```

## TODO
- [ ] object detection training & performance checking on certain datasets


## References
- [Original paper](https://arxiv.org/pdf/2102.04803.pdf)
- [Official implementation](https://github.com/xieenze/DetCo)
- [MoCo official implementation](https://github.com/facebookresearch/moco)
- [Resnet PyTorch code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
