import argparse
import os

import torch.jit
from omegaconf import OmegaConf

from train import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model to torchscript')
    parser.add_argument('-i', '--input', dest='input', default='E:/randla_opengf/outputs/ckpts/randlanet-opengf/epoch=8-valid_miou=0.8507-valid_acc=0.9459.ckpt',
                        help='Input pytorch lightning checkpoint')
    parser.add_argument('-o', '--output', dest='output', default='weights/randlanet-opengf.pt',
                        help='Output torchscript')
    parser.add_argument('-n', '--network', dest='network', default='./conf/randlanet.yaml',
                        help='path to network configuration')
    parser.add_argument('-d', '--data', dest='data', default='./conf/opengf.yaml',
                        help='path to dataset configuration')
    parser.add_argument('-t', '--train', dest='train', default='./conf/train.yaml',
                        help='path to dataset configuration')
    args = parser.parse_args()
    network = OmegaConf.load(args.network)
    data = OmegaConf.load(args.data)
    train = OmegaConf.load(os.path.abspath(args.train))
    cfg = OmegaConf.merge(data, network, train)
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    cfg.data.root = os.path.abspath(cfg.data.root)

    model = Model.load_from_checkpoint(args.input, config=cfg, map_location='cuda:0')
    script = model.to_torchscript()
    torch.jit.save(script, args.output)
