# resnet-generator
Generate Caffe Prototxt for Deep Residual Learning Network 

Only support Faster R-CNN network so far, but can be easily changed to support Fast R-CNN, classification network, etc.

Naming conventions of the layers follow the [original models](https://github.com/KaimingHe/deep-residual-networks).

## Examples

- First,  define your network in a file (see resnet50.def)

- Generate prototxt:

The script has several options, which can be listed with the `--help` flag.

Generate train/test prototxt for Faster R-CNN, 21 classes (including background):
```
./resnet_generator.py --cfg resnet50.def -t fasterrcnn --ncls 21
```

Generate train/test prototxt for Faster R-CNN, finetuning mode:
```
./resnet_generator.py --cfg resnet50_finetune.def -t fasterrcnn --ncls 21 --finetune
```

Generate train/test prototxt for Faster R-CNN, finetuning mode, fix BN parameters:
```
./resnet_generator.py --cfg resnet50_finetune.def -t fasterrcnn --ncls 21 --finetune --fixbn
```

You can also use the `--train-file`, `--test-file` flags to specify the output prototxt files.
