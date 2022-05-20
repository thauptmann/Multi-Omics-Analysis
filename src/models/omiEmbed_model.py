from abc import abstractmethod, ABC

import torch
from torch import nn


def kl_loss(mean, log_var, reduction='mean'):
    part_loss = 1 + log_var - mean.pow(2) - log_var.exp()
    if reduction == 'mean':
        loss = -0.5 * torch.mean(part_loss)
    else:
        loss = -0.5 * torch.sum(part_loss)
    return loss


class BasicModel(ABC):
    """
    This class is an abstract base class for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                          Initialize the class, first call BasicModel.__init__(self, param)
        -- <modify_commandline_parameters>:     Add model-specific parameters, and rewrite default values for existing parameters
        -- <set_input>:                         Unpack input data from the output dictionary of the dataloader
        -- <forward>:                           Get the reconstructed omics data and results for the downstream task
        -- <update>:                            Calculate losses, gradients and update network parameters
    """

    def __init__(self, param):
        """
        Initialize the BaseModel class
        """
        self.param = param
        self.gpu_ids = param.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # get device name: CPU or GPU
        self.save_dir = os.path.join(param.checkpoints_dir, param.experiment_name)
        # save all the checkpoints to save_dir, and this is where to load the models
        self.load_net_dir = os.path.join(param.checkpoints_dir, param.experiment_to_load)
        # load pretrained networks from certain experiment folder
        self.isTrain = param.isTrain
        self.phase = 'p1'
        self.epoch = 1
        self.iter = 0

        # Improve the performance if the dimensionality and shape of the input data keep the same
        torch.backends.cudnn.benchmark = True

        self.plateau_metric = 0  # used for learning rate policy 'plateau'

        self.loss_names = []
        self.model_names = []
        self.metric_names = []
        self.optimizers = []
        self.schedulers = []

        self.latent = None
        self.loss_embed = None
        self.loss_down = None
        self.loss_All = None

    @staticmethod
    def modify_commandline_parameters(parser, is_train):
        """
        Add model-specific parameters, and rewrite default values for existing parameters.
        Parameters:
            parser              -- original parameter parser
            is_train (bool)     -- whether it is currently training phase or test phase. Use this flag to add or change training-specific or test-specific parameters.
        Returns:
            The modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input_dict):
        """
        Unpack input data from the output dictionary of the dataloader
        Parameters:
            input_dict (dict): include the data tensor and its label
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Run forward pass
        """
        pass

    @abstractmethod
    def cal_losses(self):
        """
        Calculate losses
        """
        pass

    @abstractmethod
    def update(self):
        """
        Calculate losses, gradients and update network weights; called in every training iteration
        """
        pass

    def setup(self, param):
        """
        Load and print networks, create schedulers
        """
        if self.isTrain:
            self.print_networks(param)
            # For every optimizer we have a scheduler
            self.schedulers = [networks.get_scheduler(optimizer, param) for optimizer in self.optimizers]

        # Loading the networks
        if not self.isTrain or param.continue_train:
            self.load_networks(param.epoch_to_load)

    def update_learning_rate(self):
        """
        Update learning rates for all the networks
        Called at the end of each epoch
        """
        lr = self.optimizers[0].param_groups[0]['lr']

        for scheduler in self.schedulers:
            if self.param.lr_policy == 'plateau':
                scheduler.step(self.plateau_metric)
            else:
                scheduler.step()


class VaeBasicModel(BasicModel):
    """
    This is the basic VAE model class, called by all other VAE son classes.
    """

    def __init__(self, param):
        """
        Initialize the VAE basic class.
        """
        BasicModel.__init__(self, param)
        # specify the training losses you want to print out.
        if param.omics_mode == 'abc':
            self.loss_names = ['recon_A', 'recon_B', 'recon_C', 'kl']
        if param.omics_mode == 'ab':
            self.loss_names = ['recon_A', 'recon_B', 'kl']
        elif param.omics_mode == 'b':
            self.loss_names = ['recon_B', 'kl']
        elif param.omics_mode == 'a':
            self.loss_names = ['recon_A', 'kl']
        elif param.omics_mode == 'c':
            self.loss_names = ['recon_C', 'kl']
        # specify the models you want to save to the disk and load.
        self.model_names = ['Embed', 'Down']

        # input tensor
        self.input_omics = []
        self.data_index = None  # The indexes of input data

        # output tensor
        self.z = None
        self.recon_omics = None
        self.mean = None
        self.log_var = None

        # define the network
        self.netEmbed = networks.define_VAE(param.net_VAE, param.omics_dims, param.omics_mode,
                                            param.norm_type, param.filter_num, param.conv_k_size, param.leaky_slope,
                                            param.dropout_p, param.latent_space_dim, param.init_type, param.init_gain,
                                            self.gpu_ids)

        self.loss_recon_A = None
        self.loss_recon_B = None
        self.loss_recon_C = None
        self.loss_recon = None
        self.loss_kl = None

        if self.isTrain:
            # Set the optimizer
            # netEmbed and netDown can set to different initial learning rate
            self.optimizer_Embed = torch.optim.Adam(self.netEmbed.parameters(), lr=param.lr, betas=(param.beta1, 0.999), weight_decay=param.weight_decay)
            # optimizer list was already defined in BaseModel
            self.optimizers.append(self.optimizer_Embed)

            self.optimizer_Down = None

    def set_input(self, input_dict):
        """
        Unpack input data from the output dictionary of the dataloader
        Parameters:
            input_dict (dict): include the data tensor and its index.
        """
        self.input_omics = []
        for i in range(0, 3):
            if i == 1 and self.param.ch_separate:
                input_B = []
                for ch in range(0, 23):
                    input_B.append(input_dict['input_omics'][1][ch].to(self.device))
                self.input_omics.append(input_B)
            else:
                self.input_omics.append(input_dict['input_omics'][i].to(self.device))

        self.data_index = input_dict['index']

    def forward(self):
        # Get the output tensor
        self.z, self.recon_omics, self.mean, self.log_var = self.netEmbed(self.input_omics)
        # define the latent
        if self.phase == 'p1' or self.phase == 'p3':
            self.latent = self.mean
        elif self.phase == 'p2':
            self.latent = self.mean.detach()

    def cal_losses(self):
        """Calculate losses"""
        # Calculate the reconstruction loss for A
        if self.param.omics_mode == 'a' or self.param.omics_mode == 'ab' or self.param.omics_mode == 'abc':
            self.loss_recon_A = self.lossFuncRecon(self.recon_omics[0], self.input_omics[0])
        else:
            self.loss_recon_A = 0
        # Calculate the reconstruction loss for B
        if self.param.omics_mode == 'b' or self.param.omics_mode == 'ab' or self.param.omics_mode == 'abc':
            if self.param.ch_separate:
                recon_omics_B = torch.cat(self.recon_omics[1], -1)
                input_omics_B = torch.cat(self.input_omics[1], -1)
                self.loss_recon_B = self.lossFuncRecon(recon_omics_B, input_omics_B)
            else:
                self.loss_recon_B = self.lossFuncRecon(self.recon_omics[1], self.input_omics[1])
        else:
            self.loss_recon_B = 0
        # Calculate the reconstruction loss for C
        if self.param.omics_mode == 'c' or self.param.omics_mode == 'abc':
            self.loss_recon_C = self.lossFuncRecon(self.recon_omics[2], self.input_omics[2])
        else:
            self.loss_recon_C = 0
        # Overall reconstruction loss
        if self.param.reduction == 'sum':
            self.loss_recon = self.loss_recon_A + self.loss_recon_B + self.loss_recon_C
        elif self.param.reduction == 'mean':
            self.loss_recon = (self.loss_recon_A + self.loss_recon_B + self.loss_recon_C) / self.param.omics_num
        # Calculate the kl loss
        self.loss_kl = kl_loss(self.mean, self.log_var, self.param.reduction)
        # Calculate the overall vae loss (embedding loss)
        # LOSS EMBED
        self.loss_embed = self.loss_recon + self.param.k_kl * self.loss_kl


    def update(self):
        if self.phase == 'p1':
            self.forward()
            self.optimizer_Embed.zero_grad()                # Set gradients to zero
            self.cal_losses()                               # Calculate losses
            self.loss_embed.backward()                      # Backpropagation
            self.optimizer_Embed.step()                     # Update weights
        elif self.phase == 'p2':
            self.forward()
            self.optimizer_Down.zero_grad()                 # Set gradients to zero
            self.cal_losses()                               # Calculate losses
            self.loss_down.backward()                       # Backpropagation
            self.optimizer_Down.step()                      # Update weights
        elif self.phase == 'p3':
            self.forward()
            self.optimizer_Embed.zero_grad()                # Set gradients to zero
            self.optimizer_Down.zero_grad()
            self.cal_losses()                               # Calculate losses
            self.loss_All.backward()                        # Backpropagation
            self.optimizer_Embed.step()                     # Update weights
            self.optimizer_Down.step()


class FCBlock(nn.Module):
    """
    Linear => Norm1D => LeakyReLU
    """
    def __init__(self, input_dim, output_dim, leaky_slope=0.2, dropout_p=0):
        """
        Construct a fully-connected block
        Parameters:
            input_dim (int)         -- the dimension of the input tensor
            output_dim (int)        -- the dimension of the output tensor
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            activation (bool)       -- need activation or not
            normalization (bool)    -- need normalization or not
            activation_name (str)   -- name of the activation function used in the FC block
        """
        super(FCBlock, self).__init__()
        # Linear
        self.fc_block = [nn.Linear(input_dim, output_dim)]
        # Norm
        # FC block doesn't support InstanceNorm1d
        norm_layer = nn.BatchNorm1d
        self.fc_block.append(norm_layer(output_dim))
        # Dropout
        if 0 < dropout_p <= 1:
            self.fc_block.append(nn.Dropout(p=dropout_p))
        # LeakyReLU
            self.fc_block.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        y = self.fc

def define_VAE(net_VAE, omics_dims, omics_mode='multi_omics', norm_type='batch', filter_num=8, kernel_size=9, leaky_slope=0.2, dropout_p=0,
               latent_dim=256, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create the VAE network
    Parameters:
        net_VAE (str)           -- the backbone of the VAE, default: conv_1d
        omics_dims (list)       -- the list of input omics dimensions
        omics_mode (str)        -- omics types would like to use in the model
        norm_type (str)         -- the name of normalization layers used in the network, default: batch
        filter_num (int)        -- the number of filters in the first convolution layer in the VAE
        kernel_size (int)       -- the kernel size of convolution layers
        leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
        latent_dim (int)        -- the dimensionality of the latent space
        init_type (str)         -- the name of our initialization method
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
        gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1
    Returns a VAE
    The default backbone of the VAE is one dimensional convolutional layer.
    The generator has been initialized by <init_net>.
    """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    net = FcVaeABC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)

    return init_net(net, init_type, init_gain, gpu_ids)



