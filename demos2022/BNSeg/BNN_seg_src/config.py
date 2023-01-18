import os


### get config for demo
class Config():
    def __init__(self,current_path):

        self.data_dir = './BNSeg/data/' 
        self.data_name = 'CamVid'
        self.batch_size = 4
        self.is_train = False # test
        self.num_work = 4
        self.gpu = 0
        self.device = 'cpu'
        self.num_gpu = 1
        self.uncertainty = 'epistemic'
        self.n_samples = 15
        self.work_path = current_path
        self.drop_rate = 0.2
        self.in_channels = 3
        self.n_classes = 12
        self.exp_dir = './BNSeg/experiments/'
        self.exp_load = 'demo'










