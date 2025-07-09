from omegaconf import DictConfig

from models.randlanet import RandlaNet

def create_segmentation_model(config: DictConfig):
    if config.network.name == 'randlanet':
        model = RandlaNet(config)
    else:
        raise NotImplementedError('The network is not implemented')
    return model
