import time
import json
import os
from time import time
from importlib_metadata import List
import matplotlib.pyplot as plt
from os.path import join
import torch
import torch.nn as nn
import pandas

def try_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False

def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) 
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) 
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


class EpochLogger:
    def __init__(self, logger_dir, model_name):
        try_mkdir(logger_dir)
        try_mkdir(join(logger_dir, model_name))
        run_num = 0
        while os.path.exists(join(logger_dir, model_name, 'run{}'.format(run_num))):
            run_num += 1
        self.save_path = join(logger_dir, model_name, 'run{}'.format(run_num))
        try_mkdir(self.save_path)
        self.epoch_dict = dict()
        self.start_time = time()
        self.plot_object = ['test_avg_r']
        self.x_name = 'Epoch'
        self.saver_model = None
    
    def log_vars(self, vars=dict()):
        config_json = convert_json(vars)
        print(json.dumps(config_json, skipkeys=False ,separators=(',',':\t'), indent=4, sort_keys=True))
        with open(join(self.save_path, 'config.json'), 'w') as f:
            f.write(json.dumps(config_json, skipkeys=False ,separators=(',',':\t'), indent=4, sort_keys=True))

    def setup_pytorch_saver(self, model):
        assert isinstance(model, list) or isinstance(model, nn.Module)
        if isinstance(model, list):
            for module in model:
                assert isinstance(module, nn.Module)
        self.saver_model = model

    def save_model(self):
        if self.saver_model == None:
            return
        assert isinstance(self.saver_model, List) or isinstance(self.saver_model, nn.Module)
        if isinstance(self.saver_model, nn.Module):
            torch.save(self.save_model.state_dict(), join(self.save_path, 'model.pth'))
        else:
            for idx, m in zip(range(len(self.saver_model)), self.saver_model):
                torch.save(m.state_dict(), join(self.save_path, 'agent{}.pth'.format(idx)))

    def set_plot_object(self, x_name, *kwargs):
        self.x_name = x_name
        self.plot_object = kwargs


    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        # print(type(kwargs))
        for k,v in kwargs.items():
            # print(type(k), type(v))
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def plot(self):
        plt.figure()
        for k in self.plot_object:
            plt.plot(self.epoch_dict[self.x_name], self.epoch_dict[k], label=k)
        plt.legend()
        plt.savefig(join(self.save_path, 'result'))
        plt.close()

    def logging(self):
        self.store(time=int(time()-self.start_time))
        for k,v in self.epoch_dict.items():
            print("{} = {}".format(k, v[-1]))
        print('-'*100)
        dt = pandas.DataFrame(self.epoch_dict)
        dt.to_csv(join(self.save_path, 'progress.csv'))
        if len(self.plot_object) != 0:
            self.plot()
        self.save_model()
