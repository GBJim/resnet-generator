#!/usr/bin/env python
"""
Generate prototxt of the Deep Residule Learning for Caffe.
MSRA Paper: http://arxiv.org/pdf/1512.03385v1.pdf
"""
import argparse
import sys
import numpy as np

def parse_args():
    """Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate ResNet Prototxt')
    parser.add_argument('--cfg', dest='cfg_file', default=None, type=str,
                        help='Definition of the network')
    parser.add_argument('-t', '--type', dest='type', default='fasterrcnn', type=str,
                        help='Network for fasterrcnn, fastrcnn, or classification')
    parser.add_argument('--finetune', dest='finetune', action='store_true',
                        help='finetuning mode')
    parser.add_argument('--fixbn', dest='fix_bn', action='store_true',
                        help='Fix parameters of BN layers, only effective for finetuning mode')
    parser.add_argument('--ncls', dest='num_classes', default=21, type=int,
                        help='Number of categories (including background)')
    parser.add_argument('--train-file', dest='train_file', default='train.prototxt', type=str,
                        help='Output training prototxt file')
    parser.add_argument('--test-file', dest='test_file', default='test.prototxt', type=str,
                        help='Output testing prototxt file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Read network definition
    with open(args.cfg_file, 'r') as f:
        lines = f.readlines()
    sid = 0
    lid = 0
    for line in lines:
        line = line.strip()
        # print line
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        if lid == 0:
            args.num_layers = int(line)
            lid = lid + 1
            continue
        if lid == 1:
            args.num_stages = int(line)
            lid = lid + 1
            args.block_number = np.zeros((args.num_stages,1), dtype=np.int32)
            args.lr_mult = np.zeros((args.num_stages,1), dtype=np.int32)
            args.conv_params = [[] for _ in xrange(args.num_stages)]
            continue

        item = line.split(' ')
        print item
        if item[0] == 'conv':
            args.block_number[sid] = item[1]
            args.lr_mult[sid] = item[2]
            conv_param = np.zeros((0,2), dtype=np.int32)
            sid = sid + 1
        else:
            conv_param = np.vstack((conv_param, np.array([int(item[0]), int(item[1])])))
            args.conv_params[sid-1] = conv_param

    # print args.block_number, args.lr_mult, args.conv_params
    return args

def generate_data_layer(num_layers, num_classes):
    data_layer_str = '''name: "ResNet-%d"
layer {
  name: 'input-data'
  type: 'Python'
  top: 'data'
  top: 'im_info'
  top: 'gt_boxes'
  python_param {
    module: 'roi_data_layer.layer'
    layer: 'RoIDataLayer'
    param_str: "'num_classes': %d"
  }
}\n'''%(num_layers, num_classes)
    return data_layer_str

def generate_data_layer_deploy(num_layers):
    data_layer_str = '''name: "ResNet-%d"
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 224
  dim: 224
}
input: "im_info"
input_shape {
  dim: 1
  dim: 3
}\n'''%(num_layers)
    return data_layer_str

def generate_conv_layer_deploy(name, bottom, top, num_output, kernel_size, stride, bias_term):
    pad = (kernel_size - 1) / 2
    if bias_term:
        bias = 'true'
    else:
        bias = 'false'

    conv_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Convolution"
  convolution_param {
    num_output: %d
    kernel_size: %d
    pad: %d
    stride: %d
    bias_term: %s
  }
}\n'''%(name, bottom, top, num_output, kernel_size, pad, stride, bias)
    return conv_layer_str

def generate_conv_layer(name, bottom, top, lr_mult, decay_mult, lr_mult2, decay_mult2, num_output, kernel_size, stride, filler='msra', std=0.01):
    pad = (kernel_size - 1) / 2
    conv_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Convolution"
  param { lr_mult: %d decay_mult: %d }
  param { lr_mult: %d decay_mult: %d }
  convolution_param {
    num_output: %d
    kernel_size: %d
    pad: %d
    stride: %d
    weight_filler { type: "%s" std: %.3f }
    bias_filler { type: "constant" value: 0 }
  }
}\n'''%(name, bottom, top, lr_mult, decay_mult, lr_mult2, decay_mult2, num_output, kernel_size, pad, stride, filler, std)
    return conv_layer_str

def generate_conv_layer_no_bias(name, bottom, top, lr_mult, decay_mult, num_output, kernel_size, stride, filler='msra', std=0.01):
    pad = (kernel_size - 1) / 2
    conv_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Convolution"
  param { lr_mult: %d decay_mult: %d }
  convolution_param {
    num_output: %d
    kernel_size: %d
    pad: %d
    stride: %d
    bias_term: false
    weight_filler { type: "%s" std: %.3f }
  }
}\n'''%(name, bottom, top, lr_mult, decay_mult, num_output, kernel_size, pad, stride, filler, std)
    return conv_layer_str

def generate_pooling_layer(name, bottom, top, pool_type, kernel_size, stride):
    pool_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Pooling"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}\n'''%(name, bottom, top, pool_type, kernel_size, stride)
    return pool_layer_str

def generate_fc_layer(name, bottom, top, num_output, filler="msra", std=0.01):
    fc_layer_str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param { lr_mult: 1 decay_mult: 1 }
  param { lr_mult: 2 decay_mult: 0 }
  inner_product_param {
     num_output: %d
     weight_filler { type: "%s" std: %.3f }
     bias_filler { type: "constant" value: 0 }
  }
}\n'''%(name, bottom, top, num_output, filler, std)
    return fc_layer_str

def generate_activation_layer(name, bottom, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "%s"
}\n'''%(name, bottom, bottom, act_type)
    return act_layer_str

def generate_softmax_layer(name, bottom, top):
    softmax_layer_str = '''layer {
  name: "%s"
  type: "Softmax"
  bottom: "%s"
  top: "%s"
}\n'''%(name, bottom, top)
    return softmax_layer_str

def generate_softmax_loss(name, bottom0, bottom1, top):
    softmax_loss_str = '''layer {
  name: "%s"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "%s"
  propagate_down: 1
  propagate_down: 0
  top: "%s"
  loss_weight: 1
}\n'''%(name, bottom0, bottom1, top)
    return softmax_loss_str

def generate_smoothl1_loss(name, bottom, top):
    smoothl1_loss_str = '''layer {
  name: "%s"
  type: "SmoothL1Loss"
  bottom: "%s_pred"
  bottom: "%s_targets"
  bottom: "%s_inside_weights"
  bottom: "%s_outside_weights"
  top: "%s"
  loss_weight: 1
}\n'''%(name, bottom, bottom, bottom, bottom, top)
    return smoothl1_loss_str

def generate_bn_layer_deploy(bn_name, scale_name, bottom):
    bn_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "BatchNorm"
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Scale"
  scale_param {
    bias_term: true
  }
}\n'''%(bn_name, bottom, bottom, scale_name, bottom, bottom)
    return bn_layer_str

def generate_bn_layer(bn_name, scale_name, bottom, use_global_stats=True):
    # use_global_stats: set true in testing, false otherwise.
    if use_global_stats:
        ugs = 'true'
        lr_mult = 0
    else:
        ugs = 'false'
        lr_mult = 1

    bn_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "BatchNorm"
  batch_norm_param {
    use_global_stats: %s
  }
  param { lr_mult: 0 }
  param { lr_mult: 0 }
  param { lr_mult: 0 }
}
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Scale"
  scale_param {
    bias_term: true
  }
  param { lr_mult: %d }
}\n'''%(bn_name, bottom, bottom, ugs, scale_name, bottom, bottom, lr_mult)
    return bn_layer_str

def generate_eltwise_layer(name, bottom0, bottom1, top):
    eltwise_layer_str = '''layer {
	name: "%s"
	bottom: "%s"
	bottom: "%s"
	top: "%s"
	type: "Eltwise"
}\n'''%(name, bottom0, bottom1, top)
    return eltwise_layer_str

def generate_rpn_layers(bottom, num_output):
    rpn_layer_str='''layer {
  name: "rpn_conv/3x3"
  type: "Convolution"
  bottom: "%s"
  top: "rpn/output"
  param { lr_mult: 1.0 decay_mult: 1.0 }
  param { lr_mult: 2.0 decay_mult: 0 }
  convolution_param {
    num_output: %d
    kernel_size: 3 pad: 1 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_relu/3x3"
  type: "ReLU"
  bottom: "rpn/output"
  top: "rpn/output"
}
layer {
  name: "rpn_cls_score"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_cls_score"
  param { lr_mult: 1.0 decay_mult: 1.0 }
  param { lr_mult: 2.0 decay_mult: 0 }
  convolution_param {
    num_output: 18   # 2(bg/fg) * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_bbox_pred"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_bbox_pred"
  param { lr_mult: 1.0 decay_mult: 1.0 }
  param { lr_mult: 2.0 decay_mult: 0 }
  convolution_param {
    num_output: 36   # 4 * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
   bottom: "rpn_cls_score"
   top: "rpn_cls_score_reshape"
   name: "rpn_cls_score_reshape"
   type: "Reshape"
   reshape_param { shape { dim: 0 dim: 2 dim: -1 dim: 0 } }
}\n'''%(bottom, num_output)
    return rpn_layer_str

def generate_rpn_loss(feat_stride=16):
    rpn_loss_str = '''layer {
  name: 'rpn-data'
  type: 'Python'
  bottom: 'rpn_cls_score'
  bottom: 'gt_boxes'
  bottom: 'im_info'
  bottom: 'data'
  top: 'rpn_labels'
  top: 'rpn_bbox_targets'
  top: 'rpn_bbox_inside_weights'
  top: 'rpn_bbox_outside_weights'
  python_param {
    module: 'rpn.anchor_target_layer'
    layer: 'AnchorTargetLayer'
    param_str: "'feat_stride': %d"
  }
}
layer {
  name: "rpn_loss_cls"
  type: "SoftmaxWithLoss"
  bottom: "rpn_cls_score_reshape"
  bottom: "rpn_labels"
  propagate_down: 1
  propagate_down: 0
  top: "rpn_cls_loss"
  loss_weight: 1
  loss_param {
    ignore_label: -1
    normalize: true
  }
}
layer {
  name: "rpn_loss_bbox"
  type: "SmoothL1Loss"
  bottom: "rpn_bbox_pred"
  bottom: "rpn_bbox_targets"
  bottom: 'rpn_bbox_inside_weights'
  bottom: 'rpn_bbox_outside_weights'
  top: "rpn_loss_bbox"
  loss_weight: 1
  smooth_l1_loss_param { sigma: 3.0 }
}\n'''%(feat_stride)
    return rpn_loss_str

def generate_roi_layers_deploy(feat_stride=16):
    roi_layer_str = '''layer {
  name: "rpn_cls_prob"
  type: "Softmax"
  bottom: "rpn_cls_score_reshape"
  top: "rpn_cls_prob"
}
layer {
  name: 'rpn_cls_prob_reshape'
  type: 'Reshape'
  bottom: 'rpn_cls_prob'
  top: 'rpn_cls_prob_reshape'
  reshape_param { shape { dim: 0 dim: 18 dim: -1 dim: 0 } }
}
layer {
  name: 'proposal'
  type: 'Python'
  bottom: 'rpn_cls_prob_reshape'
  bottom: 'rpn_bbox_pred'
  bottom: 'im_info'
  top: 'rois'
  python_param {
    module: 'rpn.proposal_layer'
    layer: 'ProposalLayer'
    param_str: "'feat_stride': %d"
  }
}\n'''%(feat_stride)
    return roi_layer_str

def generate_roi_layers_train(num_classes, feat_stride=16):
    roi_layer_str = '''layer {
  name: "rpn_cls_prob"
  type: "Softmax"
  bottom: "rpn_cls_score_reshape"
  top: "rpn_cls_prob"
}
layer {
  name: 'rpn_cls_prob_reshape'
  type: 'Reshape'
  bottom: 'rpn_cls_prob'
  top: 'rpn_cls_prob_reshape'
  reshape_param { shape { dim: 0 dim: 18 dim: -1 dim: 0 } }
}
layer {
  name: 'proposal'
  type: 'Python'
  bottom: 'rpn_cls_prob_reshape'
  bottom: 'rpn_bbox_pred'
  bottom: 'im_info'
  top: 'rpn_rois'
  python_param {
    module: 'rpn.proposal_layer'
    layer: 'ProposalLayer'
    param_str: "'feat_stride': %d"
  }
}
layer {
  name: 'roi-data'
  type: 'Python'
  bottom: 'rpn_rois'
  bottom: 'gt_boxes'
  top: 'rois'
  top: 'labels'
  top: 'bbox_targets'
  top: 'bbox_inside_weights'
  top: 'bbox_outside_weights'
  python_param {
    module: 'rpn.proposal_target_layer'
    layer: 'ProposalTargetLayer'
    param_str: "'num_classes': %d"
  }
}\n'''%(feat_stride, num_classes)
    return roi_layer_str

def generate_roi_pooling_layer(name, bottom, top, pooled_w, pooled_h, spatial_scale=0.0625):
    roi_pooling_layer_str = '''layer {
  name: "%s"
  type: "ROIPooling"
  bottom: "%s"
  bottom: "rois"
  top: "%s"
  roi_pooling_param {
    pooled_w: %d
    pooled_h: %d
    spatial_scale: %f
  }
}\n'''%(name, bottom, top, pooled_w, pooled_h, spatial_scale)
    return roi_pooling_layer_str

def generate_train(args):
    fix_bn = args.fix_bn
    decay_mult_b = 0
    if args.finetune:
        lr_mult = 0
        decay_mult = 0
        lr_mult_b = 0
    else:
        # train
        lr_mult = 1
        decay_mult = 1
        lr_mult_b = 2
        fix_bn = False

    network_str = generate_data_layer(args.num_layers, args.num_classes)
    '''conv1'''
    filler = 'msra'
    last_top = 'data'
    network_str += generate_conv_layer('conv1', 'data', 'conv1', lr_mult, decay_mult, lr_mult_b, decay_mult_b, 64, 7, 2, filler)
    network_str += generate_bn_layer('bn_conv1', 'scale_conv1', 'conv1', fix_bn)
    network_str += generate_activation_layer('conv1_relu', 'conv1', 'ReLU')
    network_str += generate_pooling_layer('pool1', 'conv1', 'pool1', 'MAX', 3, 2)
    '''conv before ROI'''
    last_top = 'pool1'
    last_stage_top = 'pool1'
    last_output = 64
    roi_layer = len(args.block_number) - 1
    fix_bn = args.fix_bn     # set True if you don't wnat to update BN parameters
    for c in xrange(0, roi_layer):
        cid = c + 2
        conv = args.conv_params[c]
        lm = args.lr_mult[c]     # lr_mult
        dm = lm                  # decay_mult
        for l in xrange(0, args.block_number[c]):
            lid = '%d%s'%(cid, chr(97+l))
            prefix = 'res%s'%lid
            for n in xrange(0, conv.shape[0]):
                suffix = 'branch2%s'%chr(97+n)
                conv_name = '%s_%s'%(prefix, suffix)
                kernel_size = conv[n][0]
                num_output = conv[n][1]
                # use stride 2 in the first conv of each stage
                stride = 2 if c > 0 and l == 0 and n == 0 else 1
                network_str += generate_conv_layer_no_bias(conv_name, last_top, conv_name, lm, dm, num_output, kernel_size, stride, filler)
                network_str += generate_bn_layer('bn%s_%s'%(lid, suffix), 'scale%s_%s'%(lid, suffix), conv_name, fix_bn)
                if n < conv.shape[0] - 1:
                    network_str += generate_activation_layer('%s_relu'%conv_name, conv_name, 'ReLU')
                    last_top = conv_name
                else:
                    # short connection
                    if last_output != num_output:
                        stride = 2 if c > 0 and l == 0 else 1
                        suffix = 'branch1'
                        conv_name_b1 = '%s_%s'%(prefix, suffix)
                        network_str += generate_conv_layer_no_bias(conv_name_b1, last_stage_top, conv_name_b1, lm, dm, num_output, 1, stride, filler)
                        network_str += generate_bn_layer('bn%s_%s'%(lid, suffix), 'scale%s_%s'%(lid, suffix), conv_name_b1, fix_bn)
                        last_stage_top = conv_name_b1

                    network_str += generate_eltwise_layer(prefix, last_stage_top, conv_name, prefix)
                    network_str += generate_activation_layer('%s_relu'%prefix, prefix, 'ReLU')
                    last_stage_top = prefix
                    last_top = prefix
                    last_output = num_output

    '''RPN layers'''
    num_output = args.conv_params[roi_layer][0][1]
    network_str += '#============== RPN ===============\n'
    network_str += generate_rpn_layers(last_top, num_output)
    feat_stride = 2**(roi_layer+1)
    network_str += generate_rpn_loss(feat_stride)
    network_str += '#============== ROI Proposal ===============\n'
    network_str += generate_roi_layers_train(args.num_classes, feat_stride)
    network_str += '#============== RCNN ===============\n'
    roi_pooling_name = 'roi_pool%d'%(roi_layer + 2)
    spatial_scale = 1.0 / feat_stride
    pool_size = 224 * spatial_scale
    network_str += generate_roi_pooling_layer(roi_pooling_name, last_top, roi_pooling_name, pool_size, pool_size, spatial_scale)
    last_top = roi_pooling_name
    last_stage_top = roi_pooling_name

    '''R-CNN layers'''
    for c in xrange(roi_layer, len(args.block_number)):
        cid = c + 2
        conv = args.conv_params[c]
        lm = args.lr_mult[c]     # lr_mult
        dm = lm             # decay_mult
        for l in xrange(0, args.block_number[c]):
            lid = '%d%s'%(cid, chr(97+l))
            prefix = 'res%s'%lid
            for n in xrange(0, conv.shape[0]):
                suffix = 'branch2%s'%chr(97+n)
                conv_name = '%s_%s'%(prefix, suffix)
                kernel_size = conv[n][0]
                num_output = conv[n][1]
                # use stride 2 in the first conv of each stage
                stride = 2 if l == 0 and n == 0 else 1
                network_str += generate_conv_layer_no_bias(conv_name, last_top, conv_name, lm, dm, num_output, kernel_size, stride, filler)
                network_str += generate_bn_layer('bn%s_%s'%(lid, suffix), 'scale%s_%s'%(lid, suffix), conv_name, fix_bn)
                if n < conv.shape[0] - 1:
                    network_str += generate_activation_layer('%s_relu'%conv_name, conv_name, 'ReLU')
                    last_top = conv_name
                else:
                    # short connection
                    if last_output != num_output:
                        stride = 2 if c > 0 and l == 0 else 1
                        suffix = 'branch1'
                        conv_name_b1 = '%s_%s'%(prefix, suffix)
                        network_str += generate_conv_layer_no_bias(conv_name_b1, last_stage_top, conv_name_b1, lm, dm, num_output, 1, stride, filler)
                        network_str += generate_bn_layer('bn%s_%s'%(lid, suffix), 'scale%s_%s'%(lid, suffix), conv_name_b1, fix_bn)
                        last_stage_top = conv_name_b1

                    network_str += generate_eltwise_layer(prefix, last_stage_top, conv_name, prefix)
                    network_str += generate_activation_layer('%s_relu'%prefix, prefix, 'ReLU')
                    last_stage_top = prefix
                    last_top = prefix
                    last_output = num_output

    '''Average Pooling'''
    network_str += generate_pooling_layer('pool5', last_top, 'pool5', 'AVE', 7, 1)
    network_str += generate_fc_layer('cls_score', 'pool5', 'cls_score', args.num_classes, filler)
    network_str += generate_fc_layer('bbox_pred', 'pool5', 'bbox_pred', args.num_classes*4, filler)
    network_str += generate_softmax_loss('loss_cls', 'cls_score', 'labels', 'loss_cls')
    network_str += generate_smoothl1_loss('loss_bbox', 'bbox', 'loss_bbox')

    return network_str

def generate_test():
    args = parse_args()
    network_str = generate_data_layer_deploy(args.num_layers)
    '''conv1'''
    last_top = 'data'
    network_str += generate_conv_layer_deploy('conv1', 'data', 'conv1', 64, 7, 2, True)
    network_str += generate_bn_layer_deploy('bn_conv1', 'scale_conv1', 'conv1')
    network_str += generate_activation_layer('conv1_relu', 'conv1', 'ReLU')
    network_str += generate_pooling_layer('pool1', 'conv1', 'pool1', 'MAX', 3, 2)
    '''conv before ROI'''
    last_top = 'pool1'
    last_stage_top = 'pool1'
    last_output = 64
    roi_layer = len(args.block_number) - 1
    for c in xrange(0, roi_layer):
        cid = c + 2
        conv = args.conv_params[c]
        for l in xrange(0, args.block_number[c]):
            lid = '%d%s'%(cid, chr(97+l))
            prefix = 'res%s'%lid
            for n in xrange(0, conv.shape[0]):
                suffix = 'branch2%s'%chr(97+n)
                conv_name = '%s_%s'%(prefix, suffix)
                kernel_size = conv[n][0]
                num_output = conv[n][1]
                # use stride 2 in the first conv of each stage
                stride = 2 if c > 0 and l == 0 and n == 0 else 1
                network_str += generate_conv_layer_deploy(conv_name, last_top, conv_name, num_output, kernel_size, stride, False)
                network_str += generate_bn_layer_deploy('bn%s_%s'%(lid, suffix), 'scale%s_%s'%(lid, suffix), conv_name)
                if n < conv.shape[0] - 1:
                    network_str += generate_activation_layer('%s_relu'%conv_name, conv_name, 'ReLU')
                    last_top = conv_name
                else:
                    # short connection
                    if last_output != num_output:
                        stride = 2 if c > 0 and l == 0 else 1
                        suffix = 'branch1'
                        conv_name_b1 = '%s_%s'%(prefix, suffix)
                        network_str += generate_conv_layer_deploy(conv_name_b1, last_stage_top, conv_name_b1, num_output, 1, stride, False)
                        network_str += generate_bn_layer_deploy('bn%s_%s'%(lid, suffix), 'scale%s_%s'%(lid, suffix), conv_name_b1)
                        last_stage_top = conv_name_b1

                    network_str += generate_eltwise_layer(prefix, last_stage_top, conv_name, prefix)
                    network_str += generate_activation_layer('%s_relu'%prefix, prefix, 'ReLU')
                    last_stage_top = prefix
                    last_top = prefix
                    last_output = num_output

    '''RPN layers'''
    num_output = args.conv_params[roi_layer][0][1]
    network_str += '#============== RPN ===============\n'
    network_str += generate_rpn_layers(last_top, num_output)
    feat_stride = 2**(roi_layer+1)
    network_str += '#============== ROI Proposal ===============\n'
    network_str += generate_roi_layers_deploy(feat_stride)
    network_str += '#============== RCNN ===============\n'
    roi_pooling_name = 'roi_pool%d'%(roi_layer + 2)
    spatial_scale = 1.0 / feat_stride
    pool_size = 224 * spatial_scale
    network_str += generate_roi_pooling_layer(roi_pooling_name, last_top, roi_pooling_name, pool_size, pool_size, spatial_scale)
    last_top = roi_pooling_name
    last_stage_top = roi_pooling_name

    '''R-CNN layers'''
    for c in xrange(roi_layer, len(args.block_number)):
        cid = c + 2
        conv = args.conv_params[c]
        for l in xrange(0, args.block_number[c]):
            lid = '%d%s'%(cid, chr(97+l))
            prefix = 'res%s'%lid
            for n in xrange(0, conv.shape[0]):
                suffix = 'branch2%s'%chr(97+n)
                conv_name = '%s_%s'%(prefix, suffix)
                kernel_size = conv[n][0]
                num_output = conv[n][1]
                # use stride 2 in the first conv of each stage
                stride = 2 if l == 0 and n == 0 else 1
                network_str += generate_conv_layer_deploy(conv_name, last_top, conv_name, num_output, kernel_size, stride, False)
                network_str += generate_bn_layer_deploy('bn%s_%s'%(lid, suffix), 'scale%s_%s'%(lid, suffix), conv_name)
                if n < conv.shape[0] - 1:
                    network_str += generate_activation_layer('%s_relu'%conv_name, conv_name, 'ReLU')
                    last_top = conv_name
                else:
                    # short connection
                    if last_output != num_output:
                        stride = 2 if c > 0 and l == 0 else 1
                        suffix = 'branch1'
                        conv_name_b1 = '%s_%s'%(prefix, suffix)
                        network_str += generate_conv_layer_deploy(conv_name_b1, last_stage_top, conv_name_b1, num_output, 1, stride, False)
                        network_str += generate_bn_layer_deploy('bn%s_%s'%(lid, suffix), 'scale%s_%s'%(lid, suffix), conv_name_b1)
                        last_stage_top = conv_name_b1

                    network_str += generate_eltwise_layer(prefix, last_stage_top, conv_name, prefix)
                    network_str += generate_activation_layer('%s_relu'%prefix, prefix, 'ReLU')
                    last_stage_top = prefix
                    last_top = prefix
                    last_output = num_output

    '''Average Pooling'''
    network_str += generate_pooling_layer('pool5', last_top, 'pool5', 'AVE', 7, 1)
    network_str += generate_fc_layer('cls_score', 'pool5', 'cls_score', args.num_classes)
    network_str += generate_fc_layer('bbox_pred', 'pool5', 'bbox_pred', args.num_classes*4)
    network_str += generate_softmax_layer('cls_prob', 'cls_score', 'cls_prob')

    return network_str


# TODO: ResNet for classification
def generate_train_cls():
    args = parse_args()
    network_str = generate_data_layer(args.num_layers, args.num_classes)
    '''conv1'''
    last_top = 'data'
    network_str += generate_conv_layer('conv1', 'data', 'conv1', 0, 0, 0, 0, 64, 7, 2)
    network_str += generate_bn_layer('bn_conv1', 'scale_conv1', 'conv1', True)
    network_str += generate_activation_layer('conv1_relu', 'conv1', 'ReLU')
    network_str += generate_pooling_layer('pool1', 'conv1', 'pool1', 'MAX', 3, 2)
    '''conv before ROI'''
    last_top = 'pool1'
    last_stage_top = 'pool1'
    last_output = 64
    for c in xrange(0, len(args.block_number)):
        cid = c + 2
        conv = args.conv_params[c]
        lm = args.lr_mult[c]     # lr_mult
        dm = lm             # decay_mult
        for l in xrange(0, args.block_number[c]):
            lid = '%d%s'%(cid, chr(97+l))
            prefix = 'res%s'%lid
            for n in xrange(0, conv.shape[0]):
                suffix = 'branch2%s'%chr(97+n)
                conv_name = '%s_%s'%(prefix, suffix)
                kernel_size = conv[n][0]
                num_output = conv[n][1]
                # use stride 2 in the first conv of each stage
                stride = 2 if c > 0 and l == 0 and n == 0 else 1
                network_str += generate_conv_layer_no_bias(conv_name, last_top, conv_name, lm, dm, num_output, kernel_size, stride)
                network_str += generate_bn_layer('bn%s_%s'%(lid, suffix), 'scale%s_%s'%(lid, suffix), conv_name, True)
                if n < conv.shape[0] - 1:
                    network_str += generate_activation_layer('%s_relu'%conv_name, conv_name, 'ReLU')
                    last_top = conv_name
                else:
                    # short connection
                    if last_output != num_output:
                        stride = 2 if c > 0 and l == 0 else 1
                        suffix = 'branch1'
                        conv_name_b1 = '%s_%s'%(prefix, suffix)
                        network_str += generate_conv_layer_no_bias(conv_name_b1, last_stage_top, conv_name_b1, lm, dm, num_output, 1, stride)
                        network_str += generate_bn_layer('bn%s_%s'%(lid, suffix), 'scale%s_%s'%(lid, suffix), conv_name_b1, True)
                        last_stage_top = conv_name_b1

                    network_str += generate_eltwise_layer(prefix, last_stage_top, conv_name, prefix)
                    network_str += generate_activation_layer('%s_relu'%prefix, prefix, 'ReLU')
                    last_stage_top = prefix
                    last_top = prefix
                    last_output = num_output

    '''Average Pooling'''
    network_str += generate_pooling_layer('pool5', last_top, 'pool5', 'AVE', 7, 1)
    network_str += generate_fc_layer('cls_score', 'pool5', 'cls_score', args.num_classes, 'gaussian')
    network_str += generate_fc_layer('bbox_pred', 'pool5', 'bbox_pred', args.num_classes*4, 'gaussian')
    network_str += generate_softmax_loss('loss_cls', 'cls_score', 'labels', 'loss_cls')
    network_str += generate_smoothl1_loss('loss_bbox', 'bbox', 'loss_bbox')

    return network_str

def generate_solver(train_val_name):
    solver_str = '''net: "%s"
test_iter: 1000
test_interval: 6000
test_initialization: false
display: 60
base_lr: 0.1
lr_policy: "multistep"
stepvalue: 300000
stepvalue: 500000
gamma: 0.1
max_iter: 600000
momentum: 0.9
weight_decay: 0.0001
snapshot: 6000
snapshot_prefix: "pku_resnet"
solver_mode: GPU
device_id: [0,1,6,8]'''%(train_val_name)
    return solver_str

def main():
    args = parse_args()
    if args.type == 'fasterrcnn':
        train_pt = generate_train(args)
        test_pt = generate_test()
#    elif args.type == 'cls':
#        train_pt = generate_train_cls()
    else:
        print 'not available type.'

    fp = open(args.train_file, 'w')
    fp.write(train_pt)
    fp.close()
    fp = open(args.test_file, 'w')
    fp.write(test_pt)
    fp.close()

if __name__ == '__main__':
    main()
