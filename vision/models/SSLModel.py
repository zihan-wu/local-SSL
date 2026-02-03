import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, _resnet
import os 

from vision.models import Encoders
from vision.models import SSLLoss






class ContrastiveVision(torch.nn.Module):
    def __init__(self, opt, calc_loss):
        super().__init__()
        self.opt = opt
        self.contrastive_samples = self.opt.negative_samples
        if self.opt.current_rep_as_negative:
            print("Contrasting against current representation (i.e. only one negative sample)")
        else:
            print("Contrasting against ", self.contrastive_samples, " negative sample(s)")
        self.calc_loss = calc_loss
        self.encoder_type = self.opt.encoder_type
        print("Using ", self.encoder_type, " encoder")
        self.predict_module_num = self.opt.predict_module_num
        self.encoder, h_dims = self.create_encoder()
        print('Network dimensions {}'.format(h_dims))
        if calc_loss:
            if opt.proj_2d:
                self.loss = SSLLoss.CP2DLoss(opt, h_dims)
            elif opt.contrast_mode in ['infonce', 'info2nce']:
                self.loss = SSLLoss.InfoNCE(opt, h_dims)
            else:
                self.loss = SSLLoss.CPLoss(opt, h_dims)

            if opt.decode_loss == 'decode':
                self.loss_decode = SSLLoss.DecodeLoss(opt, h_dims)
            else:
                self.loss_decode = None

            if opt.extra_lateral_loss:
                self.loss_lateral = SSLLoss.CPLoss(opt, h_dims, fb_idx=range(opt.model_splits))
        self.ete_training = self.opt.ete_training
            

    def create_encoder(self):
        if self.encoder_type=='vgg_like':
            encoder, h_dims = self._create_full_model_vgg(self.opt)
        elif self.encoder_type=='resnet_like':
            encoder, h_dims = self._create_full_model_resnet(self.opt)
        elif self.encoder_type=='vgg_net':
            encoder, h_dims = self._create_vgg_net(self.opt)
        elif self.encoder_type=='scff':
            encoder, h_dims = self._create_full_model_scff(self.opt)
        else:
            raise Exception("Invalid encoder option")

        return encoder, h_dims
    
    
    def get_h_dims(self, encoder, input_dims):
        h_dims = []
        if self.opt.dataset in ['cifar10', 'cifar100']:
            img_dim = 32
        elif self.opt.dataset == 'imagenet':
            img_dim = 224
        else:
            img_dim = self.opt.random_crop_size

        return (2, input_dims, img_dim, img_dim)
        
        
        # x = torch.rand((2, input_dims, img_dim, img_dim))
        # for idx, module in enumerate(encoder):
        #     if self.opt.encoder_type == 'resnet_like':
        #         x = module(x)
        #     else:
        #         x, _ = module(x)
        #     h_dims.append(x.shape)
        # if self.opt.ete_training:
        #     h_dims = [h_dims[-1]] # only return the last module output, as this is the one used for training

        # return h_dims
    
    def _create_full_model_vgg(self, opt):

        arch = [128, 256, 'M', 256, 512, 'M', 1024, 'M', 1024, 'M']
        #arch = [16, 32, 'M', 32, 64, 'M', 128, 'M', 128, 'M']

        if opt.model_splits == 1:
            blocks = [arch]
            if opt.only_first_layers:
                blocks = [[128]]
        elif opt.model_splits == 2:
            blocks = [arch[:4], arch[4:]]
        elif opt.model_splits == 3: 
            blocks = [arch[:3], arch[3:6], arch[6:]]
        elif opt.model_splits == 6:
            blocks = [arch[:1], arch[1:3], arch[3:4], arch[4:6], arch[6:8], arch[8:]]
            #blocks = [[128, 'M'], arch[1:3], arch[3:4], arch[4:6], arch[6:8], arch[8:]]
            if opt.adjusted_baseline:
                blocks = [[128], [256, 'M'], [512, 'M'], [512], [1024, 'M'], [1024]]
            elif opt.remove_final_maxpool:
                blocks[-1] = arch[8:9]
        elif opt.model_splits == 8:
            blocks = [[128], [256, 'M'],  [512, 'M'], [512], [1024, 'M'], [1024, 'M'], [1024], [1024, 'M']]
            #blocks = [[64, "M"], [128, "M"], [256], [256, "M"], [512], [512, "M"], [512], [512, "M"]]
        else:
            raise NotImplementedError

        if opt.dataset == 'mnist':
            arch = [128, 'M', 256, 'M']
            blocks = [arch[:2], arch[2:4]]
            assert opt.model_splits == len(blocks), 'length of block not matching split'

        encoder = nn.ModuleList([])

        if opt.grayscale:
            input_dims = 1
            if opt.dataset == 'imagenet':
                input_dims = 3
        else:
            input_dims = 3


        # for idx, _ in enumerate(blocks):
        #     if idx==0:
        #         in_channels = input_dims
        #     else:
        #         in_channels = blocks[idx-1][-2] if blocks[idx-1][-1] == 'M' else blocks[idx-1][-1]


        #     encoder.append(
        #         Encoders.VGG(opt,
        #         idx,
        #         blocks,
        #         in_channels,
        #         calc_loss=False,
        #         )
        #     )

        # h_dims = self.get_h_dims(encoder, input_dims)
        
        x_dim = self.get_h_dims(encoder=None, input_dims=input_dims)
        h_dims = []
        for idx, _ in enumerate(blocks):
            if idx==0:
                in_channels = input_dims
            else:
                in_channels = blocks[idx-1][-2] if blocks[idx-1][-1] == 'M' else blocks[idx-1][-1]

            encoder.append(
                Encoders.VGG(opt,
                idx,
                blocks,
                in_channels,
                x_dim,
                calc_loss=False,
                )
            )
            x_dim = encoder[-1].get_output_dim()
            h_dims.append(x_dim)

        return encoder, h_dims
    

    def _create_vgg_net(self, opt):

        arch = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

        if opt.model_splits == 1:
            blocks = [arch]
            if opt.only_first_layers:
                blocks = [[128]]
        elif opt.model_splits == 8:
            blocks = [[64, "M"], [128, "M"], [256], [256, "M"], [512], [512, "M"], [512], [512, "M"]]
        else:
            raise NotImplementedError


        encoder = nn.ModuleList([])

        if opt.grayscale:
            input_dims = 1
            if opt.dataset == 'imagenet':
                input_dims = 3
        else:
            input_dims = 3

        x_dim = self.get_h_dims(encoder=None, input_dims=input_dims)
        h_dims = []
        for idx, _ in enumerate(blocks):
            if idx==0:
                in_channels = input_dims
            else:
                in_channels = blocks[idx-1][-2] if blocks[idx-1][-1] == 'M' else blocks[idx-1][-1]

            encoder.append(Encoders.VGG(opt,
                            idx,
                            blocks,
                            in_channels,
                            x_dim,
                            calc_loss=False,
                            ))
            x_dim = encoder[-1].get_output_dim()
            h_dims.append(x_dim)

        return encoder, h_dims

    
    def _create_full_model_resnet(self, opt):  
        assert not opt.grayscale, 'ResNet does not support grayscale input'

        if opt.resnet_num == 18:
            resnet_ = _resnet(BasicBlock, [2, 2, 2, 2], weights=None, progress=True)
        elif opt.resnet_num == 34:
            resnet_ = _resnet(BasicBlock, [3, 4, 6, 3], weights=None, progress=True)
        elif opt.resnet_num == 50:
            resnet_ = _resnet(Bottleneck, [3, 4, 6, 3], weights=None, progress=True)
        else:
            raise NotImplementedError('ResNet with {} layers not implemented'.format(opt.resnet_num))
        layer1 = nn.Sequential(
            resnet_.conv1,
            resnet_.bn1,
            resnet_.relu,
            resnet_.maxpool
        )
        encoder = nn.ModuleList([layer1, resnet_.layer1, resnet_.layer2, resnet_.layer3, resnet_.layer4])
        assert len(encoder) == opt.model_splits, 'number of encoder blocks does not match model splits'


        h_dims = self.get_h_dims(encoder, 3)
        return encoder, h_dims
    
    def _create_full_model_scff(self, opt):

        if opt.dataset == 'mnist':
            raise NotImplementedError
        elif opt.dataset == 'cifar10':
            arch = [96, 'M', 384, 'M', 1536, 'A']
            if opt.model_splits == 3: 
                blocks = [arch[:2], arch[2:4], arch[4:]]
                kernels = [5, 3, 3]
                strides = [1, 1, 1]
                paddings = [1, 1, 0]
                acts = ['triangle', 'triangle', 'relu']
            else:
                raise NotImplementedError

        encoder = nn.ModuleList([])

        if opt.grayscale:
            input_dims = 1
        else:
            input_dims = 3

        for idx, _ in enumerate(blocks):
            if idx==0:
                in_channels = input_dims
            else:
                in_channels = blocks[idx-1][0]


            encoder.append(
                Encoders.SCFF(opt,
                idx,
                blocks,
                in_channels,
                kernels[idx],
                strides[idx],
                paddings[idx],
                calc_loss=False,
                act=acts[idx]
                )
            )

        h_dims = self.get_h_dims(encoder, input_dims)
        return encoder, h_dims
    
    def set_loss_params(self, trainable=True):
        for param in self.loss.parameters():
            param.requires_grad = trainable

    def set_encoder_params(self, trainable=True):
        for param in self.encoder.parameters():
            param.requires_grad = trainable

    def forward(self, batch, label, n=6, eval=False, loss_idx_range=None, module_layer=-1, just_fb_grad=False):
        if just_fb_grad:
            fb_loss, _, _ = self.forward_fb_with_grad(batch, label, n, False, loss_idx_range, module_layer)
        # n: until which module to perform the forward pass
        model_input = batch if eval else torch.vstack([batch[0], batch[1]])
        img = model_input
        outs = []
        loss_outs = []
        # print('model input shape {}'.format(model_input.shape))
        # print('model input mean {}, std {}'.format(model_input.mean(), model_input.std()))

        if n==-1: # return (reshaped/flattened) input image, for direct classification
            s = model_input.shape # b, in_channels, y, x
            h = model_input.reshape(s[0], s[1]*s[2]*s[3]).unsqueeze(-1).unsqueeze(-1) # b, in_channels*y*x
            z = h
        else:
            # forward loop through modules
            for idx, module in enumerate(self.encoder[:n]):
                # block gradient of h at some point -> should be blocked after one module since input was detached
                if self.opt.encoder_type == 'resnet_like':
                    h = module(model_input)
                    z = h
                elif idx == n-1:
                    h, z = module(
                        model_input, label, eval=eval, module_layer=module_layer
                    )
                    
                else:
                    h, z = module(
                        model_input, label, eval=eval
                    )
                # detach z to make sure no gradients are flowing in between modules
                # we can detach z here, as for the CPC model the loop is only called once and h is forward-propagated
                if self.ete_training:
                    model_input = h
                else:
                    model_input = h.detach() # full module output

                outs.append(h) 
                loss_outs.append(z) # out: separate activated output for loss calculation
                
        if eval: # no need to compute clapp loss in test
            return None, outs, None


        if self.opt.dfa and (not self.opt.load_weights_for_gradient_calc): # we do not compute dfa loss when loading the weights into ete model for gradient analysis
            raise NotImplementedError
        else:
            loss, accuracies = self.evaluate_losses([loss_outs[-1]] if self.opt.ete_training else outs, loss_idx_range, n, images=img)

        if just_fb_grad:
            return loss, fb_loss, h, accuracies
        else:
            return loss, h, accuracies

    def forward_fb_with_grad(self, batch, label, n=6, eval=False, loss_idx_range=None, module_layer=-1):

        self.set_loss_params(trainable=True)
        # n: until which module to perform the forward pass
        model_input = batch if eval else torch.vstack([batch[0], batch[1]])
        img = model_input
        outs = []
        loss_outs = []
        # print('model input shape {}'.format(model_input.shape))
        # print('model input mean {}, std {}'.format(model_input.mean(), model_input.std()))

        if n==-1: # return (reshaped/flattened) input image, for direct classification
            s = model_input.shape # b, in_channels, y, x
            h = model_input.reshape(s[0], s[1]*s[2]*s[3]).unsqueeze(-1).unsqueeze(-1) # b, in_channels*y*x
            z = h
        else:
            # forward loop through modules
            for idx, module in enumerate(self.encoder[:n]):
                # block gradient of h at some point -> should be blocked after one module since input was detached
                if self.opt.encoder_type == 'resnet_like':
                    h = module(model_input)
                    z = h
                elif idx == n-1:
                    h, z = module(
                        model_input, label, eval=eval, module_layer=module_layer
                    )
                    
                else:
                    h, z = module(
                        model_input, label, eval=eval
                    )
                model_input = h # maintain gradient flow for fb gradient calculation

                outs.append(h) 
                loss_outs.append(z) # out: separate activated output for loss calculation

        if self.opt.dfa and (not self.opt.load_weights_for_gradient_calc): # we do not compute dfa loss when loading the weights into ete model for gradient analysis
            raise NotImplementedError
        else:
            # loss, accuracies = self.evaluate_losses([loss_outs[-1]] if self.opt.ete_training else outs, loss_idx_range, n, images=img)
            loss, rand_index, rand_fixation, accuracies = self.loss.forward_best_grad_pred(outs, rand_index=None, idx_range = loss_idx_range, rand_fixation=None)

        self.set_loss_params(trainable=False)
        return loss, h, accuracies

    def evaluate_losses(self, h_list, loss_idx_range, n, images): 
        # loop BACKWARDS through module outs and calculate losses
        # backward loop is necessary because of potential feedback gating!
        save_index = False
        
        if self.opt.reload_index_path is None:
            rand_index = None
            rand_fixation = None
        elif os.path.exists(self.opt.reload_index_path):
            rand_index = torch.load(self.opt.reload_index_path)
            rand_fixation = torch.load(self.opt.reload_fixation_path) if self.opt.either_pos_or_neg_update else None
            print('random index loaded from {}'.format(self.opt.reload_index_path))
        else:
            rand_index = None
            rand_fixation = None
            save_index = True

        loss, rand_index, rand_fixation, accuracies = self.loss(h_list, rand_index=rand_index, idx_range = loss_idx_range, rand_fixation=rand_fixation)
        if self.opt.extra_lateral_loss:
            loss_extra, rand_index, rand_fixation, accuracies = self.loss_lateral(h_list, rand_index=rand_index, idx_range = loss_idx_range, rand_fixation=rand_fixation)
            loss['loss'] = loss['loss'] + loss_extra['loss']

        if self.loss_decode is not None:
            loss_decode = self.loss_decode([images] + h_list)
            if self.opt.ete_training:
                raise NotImplementedError("ETE training does not support decode loss")
            else:
                loss['loss'] = loss['loss'] + self.opt.decode_loss_coeff * loss_decode
                loss['loss_decode'] = loss_decode

        if self.opt.contrain_norm:
            loss['loss_norm'] = torch.zeros(1, n, device=loss['loss'].get_device())
            #print('shape {}'.format(loss['loss'].shape))
            for idx, module in enumerate(self.encoder[:n]):
                loss_norm = module.constrain_weight()
                loss['loss_norm'][0, idx] = loss_norm
                if self.opt.ete_training:
                    loss['loss'][0, -1] = loss['loss'][0, -1] + 0.05*loss_norm
                else:
                    loss['loss'][0, idx] = loss['loss'][0, idx] + 0.05*loss_norm

        if save_index:
            #print(type(rand_index), len(rand_index), rand_index[0])
            torch.save(rand_index, self.opt.reload_index_path)
            if self.opt.either_pos_or_neg_update:
                torch.save(rand_fixation, self.opt.reload_fixation_path)

        return loss, accuracies

