####################################################################################
################# setting hyper-Parameters EfficientNet B5 #########################

from efficientnet_pytorch import utils
import collections
from collections import OrderedDict 
import re
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
from torch.nn import functional as F
from inplace_abn.abn import InPlaceABN

from efficientnet_pytorch.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]

class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


######################################################
################### MB-Blocks ########################


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = InPlaceABN(oup)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = InPlaceABN(oup)

        # Squeeze and Excitation layer, if desired

        ## Deleted it from here

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = InPlaceABN(final_oup)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._bn0(self._expand_conv(inputs))
        x = self._bn1(self._depthwise_conv(x))

        # Squeeze and Excitation
        # Delete SE layer from here


        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x



##################################################################################################################################
############################ EfficientNet Architecture ( MB blocks only it doesn't include head tail )  ##########################

class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args


        # Build blocks
        self._blocks = nn.ModuleList([])
        self._array = []
        for block_args in self._blocks_args:

            self.temp = []
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            self.temp.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))
                self.temp.append(MBConvBlock(block_args, self._global_params))
            self._array.append( nn.Sequential(*self.temp))


    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(inputs, drop_connect_rate=drop_connect_rate)
            
        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        return x
    
    ###################################################################
    ############### returning the middle blocks #######################
    def return_sub(self):
      return self._array


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class FPN(nn.Module):
     
  def __init__(self, blocks,blocks_args=None, global_params=None):
    super().__init__()

    assert isinstance(blocks_args, list), 'blocks_args should be a list'
    assert len(blocks_args) > 0, 'block args must be greater than 0'
    self._global_params = global_params
    self._blocks_args = blocks_args

    # Get static or dynamic convolution depending on image size
    Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

    # Stem
    self._conv_stem = Conv2d(3, 48, kernel_size=3, stride=2, bias=False)
    self._bn0 = InPlaceABN(48)

    #blocks
    self.blocks0 = blocks[0]
    self.blocks1 = blocks[1]
    self.blocks2 = blocks[2]
    self.blocks3 = blocks[3]
    self.blocks4 = blocks[4]
    self.blocks5 = blocks[5]
    self.blocks6 = blocks[6]

    # Head
    self._conv_head = Conv2d(512, 2048, kernel_size=1, bias=False)
    self._bn1 = InPlaceABN(2048)

    same_Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

    # upper pyramid
    self.conv_up1 = same_Conv2d(40, 256, kernel_size=1, stride=1, bias=False)
    self.conv_up2 = same_Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
    self.conv_up3 = same_Conv2d(176, 256, kernel_size=1, stride=1, bias=False)
    self.conv_up4 = same_Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)

    self.inABNone = InPlaceABN(256)
    self.inABNtwo = InPlaceABN(256)
    self.inABNthree = InPlaceABN(256)
    self.inABNfour = InPlaceABN(256)

    #separable

    self.separable1 = SeparableConv2d(256, 256, 3)
    self.separable2 = SeparableConv2d(256, 256, 3)
    self.separable3 = SeparableConv2d(256, 256, 3)
    self.separable4 = SeparableConv2d(256, 256, 3)

    self.SepinABNone = InPlaceABN(256)
    self.SepinABNtwo = InPlaceABN(256)
    self.SepinABNthree = InPlaceABN(256)
    self.SepinABNfour = InPlaceABN(256)

    # upsample bilinear

    self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
    self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
    self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')


    # downsample

    self.down1 = nn.MaxPool2d(2, stride=2)
    self.down2 = nn.MaxPool2d(2, stride=2)
    self.down3 = nn.MaxPool2d(2, stride=2)

    #additional layer
    self.additional_down = nn.MaxPool2d(2, stride=2)
    
  def forward(self, x):

      # Stem
      x = self._bn0(self._conv_stem(x))
      # Blocks
      x = self.blocks0(x)
      x1 = self.blocks1(x)
      x2 = self.blocks2(x1)
      x = self.blocks3(x2)
      x3 = self.blocks4(x)
      x = self.blocks5(x3)
      x = self.blocks6(x)

      # Head
      x4 = self._bn1(self._conv_head(x))
      
      #pyramids

      u1 = self.inABNone(self.conv_up1(x1))
      u2 = self.inABNtwo(self.conv_up2(x2))
      u3 = self.inABNthree(self.conv_up3(x3))
      u4 = self.inABNfour(self.conv_up4(x4))
      
      uu1 = self.down1(u1) 
      
      uu2 = self.down2(uu1) + u3
      uu3 = self.down3(uu2) + u4
      
      low1 = uu3 + u4
      final1 = self.SepinABNone(self.separable1(low1))

      low2 = u3 + self.up1(u4)
      final2 = self.SepinABNtwo(self.separable2(low2 + uu2))

      low3 = u2 + self.up2(low2)
      final3 = self.SepinABNthree(self.separable3(low3 + uu1))

      low4 = u1 + self.up3(low3)
      final4 = self.SepinABNfour(self.separable4(low4 + u1))

      final5 = self.additional_down(final1)
      od = OrderedDict() 
      od['0'] = final4
      od['1'] = final3
      od['2'] = final2
      od['3'] = final1
      od['4'] = final5
        
      return od

def TwoWayFPNBackbone():

    override_params={'num_classes': 1000}
    paras = get_model_params( 'efficientnet-b5', override_params )

    model = EfficientNet(paras[0],paras[1])
    arr = model.return_sub()

    fpn = FPN( arr, paras[0], paras[1] )
    return fpn