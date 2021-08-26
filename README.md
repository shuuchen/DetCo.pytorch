# DetCo.pytorch
An unofficial PyTorch implementation of DetCo [paper](https://arxiv.org/pdf/2102.04803.pdf). Official implementation can be found [here](https://github.com/xieenze/DetCo).

<img width="746" alt="スクリーンショット 2021-03-30 11 39 04" src="https://user-images.githubusercontent.com/13246825/112925323-a5cab780-914c-11eb-80f9-19199fb439d6.png">

This implementation is based on the official implementation of [MoCo](https://github.com/facebookresearch/moco) and [PyTorch resnet code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py). Though list is mentioned in original paper, this implementation use pure tensors without list, which enables parallel computing.


## Requirements
```
pip install -r requirements.txt
```

## Usage
- Please refer to [main_decto.py](https://github.com/shuuchen/DetCo.pytorch/blob/main/main_decto.py):

## Unsupervised training
As specified in the original paper, DetCo is trained with 8 GPUs with batch shuffling. However, sometimes you cannot use all the GPUs in an 8-GPU machine. I used half of them for training. If you also need to modify the number of GPUs like me, check the following two lines of code:
  - [number of GPUs to use](https://github.com/shuuchen/DetCo.pytorch/blob/main/main_detco.py#L124)
  - [master GPU number](https://github.com/shuuchen/DetCo.pytorch/blob/main/main_detco.py#L136)
  
This implementation reserved original moco implementation as much as possible. Therefore the command line for unsupervised training is also similar to that of [moco](https://github.com/facebookresearch/moco#unsupervised-training). You can train like:
  - ```python
    python main_detco.py \
      -a resnet50 \
      --lr 0.015 \
      --batch-size 128 \
      --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
      [your imagenet-folder with train and val folders] \
      --detco-t 0.2 --aug-plus --cos
    ```

## TODO
- [ ] object detection training & performance checking on certain datasets


## References
- [Original paper](https://arxiv.org/pdf/2102.04803.pdf)
- [Official implementation](https://github.com/xieenze/DetCo)
- [MoCo official implementation](https://github.com/facebookresearch/moco)
- [Resnet PyTorch code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
