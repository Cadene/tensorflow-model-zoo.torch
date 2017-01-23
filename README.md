# Tensorflow Model Zoo for Torch7 and PyTorch

This is a porting of tensorflow pretrained models made by [Remi Cadene](http://remicadene.com) and [Micael Carvalho](http://micaelcarvalho.com). It includes only InceptionV4 for now :(

This work was inspired by [inception-v3.torch](https://github.com/Moodstocks/inception-v3.torch).

Please beware of the LICENSE made up by Google before using the pretrained models.

## Requirements

- Tensorflow
- Torch7
- PyTorch
- hdf5 for python3
- hdf5 for lua

## How to use with Torch7

Please go to [torchnet-vision](https://github.com/Cadene/torchnet-vision). You will find all the pretrained models there :)

## How to reproduce 

```
python3 inceptionv4/tensorflow_dump.py
th inceptionv4/torch_load.lua
python3 inceptionv4/pytorch_load.py
```