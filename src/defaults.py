import cv2

default_process = {'gauss' : {'ksize': (5, 5), 'sigmaX': 1.5},
                                     'median': {'ksize': 5},
                                     'normalize': {'alpha': 0, 'beta': 255, 'norm_type': cv2.NORM_MINMAX},
                                     'flags' : ['laplace']}
default_flow = {'pyr_scale' : 0.5,
                                'levels' : 3,
                                'winsize' : 15,
                                'iterations' : 3,
                                'poly_n' : 5,
                                'poly_sigma' : 1.2,
                                'flag' : 0}

default_yaml_config = {
                        'preprocess_args': {
                            'gauss': {'ksize': [5, 5], 'sigmaX': 1.5}, 
                            'median': {'ksize': 5}, 
                            'normalize': {'alpha': None, 'beta': None}, 
                            'contrast': {'alpha': 1.0, 'beta': 0.0}, 
                            'skip': []
                        }, 
                        'farneback_args': {
                            'levels': 0.5, 
                            'winsize': 3, 
                            'iterations': 3, 
                            'poly_n': 5, 
                            'poly_sigma': 1.2, 
                            'flags': 0
                        }, 
                        'raft_args': {
                            'model_size': 1, 
                            'model_weights_path': '', 
                            'gpu_flag': False
                        }
                    }
