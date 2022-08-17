import yaml
from torchvision import transforms
from src import data
# from src import conv_onet
from src.config_inside import get_model


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model_outside(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''

    model = get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = conv_onet.config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = conv_onet.config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method'] # convonet
    dataset_type = cfg['data']['dataset'] # Shapes3D
    dataset_folder = cfg['data']['path'] # data/demo/synthetic_room_dataset
    categories = cfg['data']['classes'] # 

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode] # mdoe = 'test'

    # Create dataset 似乎现在只有一类dataset type: shapes3d
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = conv_onet.config.get_data_fields(mode, cfg) # {}
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField() # 这个类似乎还没有开发完

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
 
    return dataset


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type'] # input_type: pointcloud

    if input_type is None:
        inputs_field = None
    elif input_type == 'pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ]) # 这里只是初始化
        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], 
            transform,
            multi_files= cfg['data']['multi_files']
        ) # 这里只是初始化
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field