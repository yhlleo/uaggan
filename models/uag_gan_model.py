import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import uag_networks

# TODO: finish the model

class UAGGANModel(BaseModel):
    '''
      An implement of the UAGGAN model.
  
      Paper: Unsupervised Attention-guided Image-to-Image Translation, NIPS 2018.
             https://arxiv.org/pdf/1806.02311.pdf
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)

        return parser

    def __init__(self, opt):
        """Initialize the UAGGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        pass