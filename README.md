# Tensorflow Model Zoo for Torch7

This is a porting of tensorflow pretrained models made by [Remi Cadene](http://remicadene.com) and [Micael Carvalho](http://webia.lip6.fr/~carvalho). It includes only InceptionV4 for now :(

This work was inspired by [inception-v3.torch](https://github.com/Moodstocks/inception-v3.torch).

Please beware of the LICENCE made up by Google before using the pretrained models.

## Requirements

- Tensorflow
- Torch7
- hdf5 for python3
- hdf5 for lua

## How to use

Please go to [torchnet-vision](https://github.com/Cadene/torchnet-vision). You will find all the pretrained models there :)

## How to reproduce 

```
python3 inceptionv4_dump.py
th inceptionv4_load.py
```