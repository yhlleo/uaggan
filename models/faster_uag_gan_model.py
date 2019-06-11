import torch
import itertools
from util.image_pool import ImageMaskPool, ImagePool
from .base_model import BaseModel
from . import networks 
from . import uag_networks as uag

def compute_loss_smooth(mat):
    """Total Variation Regularization"""
    return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
        torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

class FasterUAGGANModel(BaseModel):
    '''
      An implement of the UAGGAN model.
  
      Paper: Unsupervised Attention-guided Image-to-Image Translation, NIPS 2018.
             https://arxiv.org/pdf/1806.02311.pdf
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
        return parser

    def __init__(self, opt):
        """Initialize the UAGGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'D_B', 'G_A', 'G_B', 'cycle_A', 'cycle_B']#, 'smooth']
        self.visual_names = ['real_A', 'att_A_viz', 'fake_B', 'masked_fake_B', 
                             'real_B', 'att_B_viz', 'fake_A', 'masked_fake_A']
        if self.isTrain:
            self.model_names = ['G_img_A', 'G_img_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_img_A', 'G_img_B']
      
        self.netG_img_A = uag.define_net_faster(opt.input_nc,
                                                opt.output_nc,
                                                opt.ngf,
                                                norm=opt.norm,
                                                init_type=opt.init_type,
                                                init_gain=opt.init_gain,
                                                gpu_ids=opt.gpu_ids)
        
        self.netG_img_B = uag.define_net_faster(opt.input_nc,
                                                opt.output_nc,
                                                opt.ngf,
                                                norm=opt.norm,
                                                init_type=opt.init_type,
                                                init_gain=opt.init_gain,
                                                gpu_ids=opt.gpu_ids)


        if self.isTrain:
            self.netD_A = uag.define_net_dis(opt.input_nc,
                                             opt.ndf,
                                             norm=opt.norm,
                                             init_type=opt.init_type,
                                             init_gain=opt.init_gain,
                                             gpu_ids=opt.gpu_ids)

            self.netD_B = uag.define_net_dis(opt.input_nc,
                                             opt.ndf,
                                             norm=opt.norm,
                                             init_type=opt.init_type,
                                             init_gain=opt.init_gain,
                                             gpu_ids=opt.gpu_ids)

        if self.isTrain:
            self.masked_fake_A_pool = ImagePool(opt.pool_size)
            self.masked_fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                    self.netG_img_A.parameters(), 
                    self.netG_img_B.parameters()), 
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                    self.netD_A.parameters(), 
                    self.netD_B.parameters()), 
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # G(A) -> B
        self.fake_B, self.att_A = self.netG_img_A(self.real_A)
        if not self.isTrain:
            self.att_A *= (self.att_A>self.opt.thresh).float()
        self.masked_fake_B = self.fake_B*self.att_A + self.real_A*(1-self.att_A)
        # G(B) -> A
        self.fake_A, self.att_B = self.netG_img_B(self.real_B)
        if not self.isTrain:
            self.att_B *= (self.att_B>self.opt.thresh).float()
        self.masked_fake_A = self.fake_A*self.att_B + self.real_B*(1-self.att_B)

        # cycle G(G(A)) -> A
        self.cycle_fake_A, self.cycle_att_B = self.netG_img_B(self.masked_fake_B)
        self.cycle_masked_fake_A = self.cycle_fake_A*self.cycle_att_B + self.masked_fake_B*(1-self.cycle_att_B)
        # cycle G(G(B)) -> B
        self.cycle_fake_B, self.cycle_att_A = self.netG_img_A(self.masked_fake_A)
        self.cycle_masked_fake_B = self.cycle_fake_B*self.cycle_att_A + self.masked_fake_A*(1-self.cycle_att_A)

        # just for visualization
        self.att_A_viz, self.att_B_viz = (self.att_A-0.5)/0.5, (self.att_B-0.5)/0.5

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        masked_fake_B = self.masked_fake_B_pool.query(self.masked_fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, masked_fake_B)

        masked_fake_A = self.masked_fake_A_pool.query(self.masked_fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, masked_fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # GAN loss D_A(G(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.masked_fake_B), True)
        # GAN loss D_B(G(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.masked_fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.cycle_masked_fake_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.cycle_masked_fake_B, self.real_B) * lambda_B

        #self.loss_smooth = compute_loss_smooth(self.att_A) + compute_loss_smooth(self.att_B)
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B #+ self.loss_smooth
        self.loss_G.backward()
    
    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()        # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights