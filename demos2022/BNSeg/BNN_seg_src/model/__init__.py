import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P

from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        print('Making model...')

        self.is_train = config.is_train
        self.device = config.device
        self.num_gpu = config.num_gpu
        self.uncertainty = config.uncertainty

        self.n_samples = config.n_samples
        if config.is_train:
            module = import_module('model.' + self.uncertainty)
        else: 
            module = import_module('BNSeg.BNN_seg_src.model.' + self.uncertainty)
        self.model = module.make_model(config).to(config.device)

    def forward(self, input,  sample_numer_e=0):
        if self.model.training:
            if self.num_gpu > 1:
                return P.data_parallel(self.model, input,
                                       list(range(self.num_gpu)))
            else:
                return self.model.forward(input)
        else:
            forward_func = self.model.forward
            if  self.uncertainty == 'combined' or self.uncertainty == 'combined_c':
                return self.test_combined(input, forward_func,sample_numer_e)

    def test_combined(self, input, forward_func, sample_numer_e= 15):
    
        def calc_entropy(input_tensor):
            lsm = nn.LogSoftmax(dim=1)
            log_probs = lsm(input_tensor) #
            probs = torch.exp(log_probs)
            p_log_p = log_probs * probs
            entropy = -p_log_p.mean(dim=1)
            return entropy
        
        if sample_numer_e == 0:
            sample_numer_e = self.n_samples

        mean_samples = []
        var_a_samples = []
        # n_samples model samples
        for i_sample in tqdm(range(sample_numer_e), desc='Sample model progress', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            results = forward_func(input)
            mean_i = results['mean']
            var_i = results['var']
            var_i = torch.exp(var_i)
            mean_samples.append(mean_i)
            var_a_samples.append(var_i)


        mean = torch.stack(mean_samples, dim=0).mean(dim=0)
 

        a_var = torch.stack(var_a_samples, dim=0).mean(dim=0) #  B X C X H x W
        a_var = torch.sqrt(a_var)
        a_var_entropy_high = calc_entropy(mean+a_var)
        a_var_entropy_low = calc_entropy(mean-a_var)
        a_var = torch.abs(a_var_entropy_high - a_var_entropy_low)

        a_var = a_var/0.2071 #a_var.max() # maximal entropy assume all output uniform distrbution

        e_var = torch.stack(mean_samples, dim=0).var(dim=0) # 
        # e_var = torch.sqrt(e_var)
        e_var_entropy_high = calc_entropy(mean+e_var)
        e_var_entropy_low = calc_entropy(mean-e_var)
        e_variance = torch.abs(e_var_entropy_high - e_var_entropy_low)

        e_variance = e_variance/0.2071 #e_variance.max()

        

        results = {'mean': mean, 'e_var': e_variance, 'a_var':a_var, 'var_a_samples':var_a_samples,
        'sample_0':mean_samples[0],'sample_1':mean_samples[1],'sample_2':mean_samples[2] }
        return results


    def save(self, ckpt, epoch):
        save_dirs = [os.path.join(ckpt.model_dir, 'model_latest.pt')]
        save_dirs.append(
            os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, ckpt, cpu=False):
        epoch = ckpt.last_epoch
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        if epoch == -1:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_latest.pt'), **kwargs)
        else:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def load(self, ckpt, cpu=False):
        epoch = ckpt.last_epoch
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        if epoch == -1:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_latest.pt'), **kwargs)
        else:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)
