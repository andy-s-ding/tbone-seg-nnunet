#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
import time

from nnunet.network_architecture.discriminator_3D import Discriminator
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler

class nnUNetTrainerV2_Disc(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.dis_threshold = 0.5
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        
        self.all_tr_ad_losses = []
        self.all_val_ad_losses = []
        self.all_val_ad_acc = []
        self.pin_memory = True

        # debugging
        self.num_batches_per_epoch = 2
        self.num_val_batches_per_epoch = 2
        self.save_every = 1

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            # Disriminator loss:
            self.loss_ad = loss_fn = nn.BCEWithLogitsLoss()

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # Init UNet
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)

        # load_pretrained_weights(self.network, args.pretrained_weights)

        # Init 3D Discriminator
        self.adversary = Discriminator(in_channels=17)
        if torch.cuda.is_available():
            self.network.cuda()
            self.adversary.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.adversary is not None, "self.initialize_network must be called first"
        self.ad_optimizer = torch.optim.SGD(self.adversary.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        ret = super().initialize_optimizer_and_scheduler()
        return ret
        # Init network optimizer
        
        

    def run_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time.time()
            
            ad_losses_epoch = []
            train_losses_epoch = []

            # train one epoch
            self.network.eval()
            self.adversary.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l_ad, _ = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l_ad)
                        
                        ad_losses_epoch.append(l_ad)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l_ad, _ = self.run_iteration(self.tr_gen, True)
                    
                    ad_losses_epoch.append(l_ad)

            self.all_tr_ad_losses.append(np.mean(ad_losses_epoch))
            self.print_to_log_file("ad loss : %.4f" % self.all_tr_ad_losses[-1])
            

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                self.adversary.eval()
                val_ad_losses = []
                val_ad_acc = []
                for b in range(self.num_val_batches_per_epoch):
                    l_ad, ad_acc = self.run_iteration(self.val_gen, False, True)
                    val_ad_losses.append(l_ad)
                    val_ad_acc.append(ad_acc)
                self.all_val_ad_losses.append(np.mean(val_ad_losses))
                self.all_val_ad_acc.append(np.mean(val_ad_acc))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_ad_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    self.adversary.train()
                    # validation with train=True
                    val_ad_losses = []
                    val_ad_acc = []
                    for b in range(self.num_val_batches_per_epoch):
                        l_ad, ad_acc = self.run_iteration(self.val_gen, False)
                        val_ad_losses.append(l_ad)
                        val_ad_acc.append(ad_acc)
                    self.all_val_ad_losses.append(np.mean(val_ad_losses))
                    self.all_val_ad_acc.append(np.mean(val_ad_acc))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_ad_losses[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time.time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))
    
    def make_probability(self, x, mean=0, stddev=2):
        
        # Make vector one hot, make correct shape.
        y = torch.nn.functional.one_hot(x.long(), 17)
        y = y.squeeze().permute(0, 4, 1, 2, 3) # This part checks out

        # Add noise w/ std 
        noised_input = y + abs(torch.autograd.Variable(torch.rand(y.shape)).cuda() + mean) / stddev
        probability = torch.nn.Softmax(dim=1)(noised_input)
        _, verify = torch.max(probability, dim=1, keepdims=True)
        # Check that label map still is accurate.
        assert(torch.all(verify == x))
        return probability

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """

        # Load Data
        dl_start = time.time()
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        
        # target always first input to discriminator, so correct label for disc is always [0, 0]. 
        # Use one hot encoding with BCEWithLogitsLoss()
        dis_target_one_hot = torch.Tensor([[1, 0], [1, 0]])
        _, dis_target = torch.max(dis_target_one_hot, dim=1)

        # network input
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            dis_target = to_cuda(dis_target)
            dis_target_one_hot = to_cuda(dis_target_one_hot)

        print("Time to load data: ", time.time() - dl_start, "seconds.")

        
        f_start = time.time()
        if self.fp16:
            
            with autocast(): 
                ## Train Discriminator
                
                # Get generator output 
                gen_pred = self.network(data)           
                del data    

                # Get discriminator output
                
                dis_output = self.adversary(self.make_probability(target[0]), gen_pred[0])
                _, dis_pred = torch.max(dis_output, dim=1)

                
                # Calculate loss
                dis_loss = self.loss_ad(dis_target_one_hot, dis_output)
                
                # Calculate accuracy
                dis_acc = (dis_pred == dis_target).sum()
                if do_backprop and (dis_acc < self.dis_threshold):  ## CHANGE dis_threshold if needed
                    self.ad_optimizer.zero_grad()
                    dis_loss.backward(retain_graph=True)
                    self.ad_optimizer.step()
                    discriminator_update = 'True'

                del gen_pred

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        
        return dis_loss.detach().cpu().numpy(), dis_acc.cpu().numpy()

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax1 = fig.add_subplot(111)  # combined
            
            ax1_2 = ax1.twinx()
            
            x_values = list(range(self.epoch + 1))

            ax1.plot(x_values, self.all_tr_ad_losses, color='b', ls='-', label="loss_tr_ad")

            ax1.plot(x_values, self.all_val_ad_losses, color='r', ls='-', label="loss_val_ad, train=False")

            if len(self.all_val_ad_acc) == len(x_values):
                ax1_2.plot(x_values, self.all_val_ad_acc, color='g', ls='--', label="acc_ad metric")  # gen

            ax1.set_xlabel("epoch")
            ax1.set_ylabel("loss")
            ax1_2.set_ylabel("evaluation metric")
            ax1.legend()
            ax1_2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()

        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time.time()
        gen_state_dict = self.network.state_dict()
        ad_state_dict = self.adversary.state_dict()
        # move params to cpu
        for key in gen_state_dict.keys():
            gen_state_dict[key] = gen_state_dict[key].cpu()
        for key in ad_state_dict.keys():
            ad_state_dict[key] = ad_state_dict[key].cpu()

        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            gen_optimizer_state_dict = self.optimizer.state_dict()
            ad_optimizer_state_dict = self.ad_optimizer.state_dict()
        else:
            optimizer_state_dict = None
            ad_optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': gen_state_dict, # called just "state_dict" for generator to make pretrained weights function compatible
            'ad_state_dict': ad_state_dict,
            'gen_optimizer_state_dict': gen_optimizer_state_dict,
            'ad_optimizer_state_dict': ad_optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_ad_losses, self.all_val_ad_losses, self.all_val_ad_acc),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time.time() - start_time))


    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_gen_state_dict = OrderedDict()
        new_ad_state_dict = OrderedDict()
        curr_gen_state_dict_keys = list(self.network.state_dict().keys())
        curr_ad_state_dict_keys = list(self.adversary.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        # do for generator
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_gen_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_gen_state_dict[key] = value
        # do for adversary
        for k, value in checkpoint['ad_state_dict'].items():
            key = k
            if key not in curr_ad_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_ad_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_gen_state_dict)
        self.adversary.load_state_dict(new_ad_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            gen_optimizer_state_dict = checkpoint['gen_optimizer_state_dict']
            if gen_optimizer_state_dict is not None:
                self.optimizer.load_state_dict(gen_optimizer_state_dict)

            ad_optimizer_state_dict = checkpoint['ad_optimizer_state_dict']
            if ad_optimizer_state_dict is not None:
                self.ad_optimizer.load_state_dict(ad_optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        # self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
        #     'plot_stuff']

        self.all_tr_ad_losses, self.all_val_ad_losses, self.all_val_ad_acc = checkpoint['plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()