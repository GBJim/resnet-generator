# resnet-generator
Generate Caffe Prototxt for Deep Residual Learning Network

## Examples

### First,  Define your network in a file (see resnet50.def)

### Generate prototxt:
Generate train/test prototxt for Faster R-CNN, 21 classes (including background):
'''
./resnet_generator.py --cfg resnet50.def -t fasterrcnn --ncls 21
'''

Generate train/test prototxt for Faster R-CNN, finetuning mode:
'''
./resnet_generator.py --cfg resnet50_finetune.def -t fasterrcnn --ncls 21 --finetune
'''

Generate train/test prototxt for Faster R-CNN, finetuning mode, fix BN parameters:
'''
./resnet_generator.py --cfg resnet50_finetune.def -t fasterrcnn --ncls 21 --finetune --fixbn
'''
