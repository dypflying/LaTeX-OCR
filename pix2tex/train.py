from pix2tex.dataset.dataset import Im2LatexDataset
import os
import argparse
import logging
import yaml

import torch
from munch import Munch
from tqdm.auto import tqdm
import wandb
import torch.nn as nn
from pix2tex.eval import evaluate
from pix2tex.models import get_model
# from pix2tex.utils import *
from pix2tex.utils import in_model_path, parse_args, seed_everything, get_optimizer, get_scheduler, gpu_memory_check


def train(args):
    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(**args, test=False)
    valdataloader = Im2LatexDataset().load(args.valdata)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    valdataloader.update(**valargs)
    device = args.device
    model = get_model(args)
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_memory_check(model, args)
    out_path = os.path.join(args.model_path, args.name)
    os.makedirs(out_path, exist_ok=True)

    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt, map_location=device))

    def save_models(epoch, step, bleu_score=0, edit_distance=0, token_accuracy=0):
        torch.save(model.state_dict(), os.path.join(out_path, '%s_epoch%02d_step%02d.pth' % (args.name, epoch+1, step+1)))
        yaml.dump(dict(args), open(os.path.join(out_path, 'config.yaml'), 'w+'))
        if bleu_score > 0 or edit_distance>0 or token_accuracy>0: 
            with open(os.path.join(out_path, 'epoch%02d_step%02d_value.txt' % (epoch+1, step+1) ), 'w+') as file:
                file.write(f"bleu_score:{bleu_score}, edit_distance:{edit_distance}, token_accuracy:{token_accuracy}")


    opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas)
    scheduler = get_scheduler(args.scheduler)(opt, step_size=args.lr_step, gamma=args.gamma)

    microbatch = args.get('micro_batchsize', -1)
    if microbatch == -1:
        microbatch = args.batchsize

    try:
        for e in range(args.epoch, args.epochs):
            args.epoch = e
            dset = tqdm(iter(dataloader))
            for i, (seq, im) in enumerate(dset):
                if seq is not None and im is not None:
                    opt.zero_grad()
                    total_loss = 0
                    for j in range(0, len(im), microbatch):
                        tgt_seq, tgt_mask = seq['input_ids'][j:j+microbatch].to(device), seq['attention_mask'][j:j+microbatch].bool().to(device)
                        loss = model.data_parallel(im[j:j+microbatch].to(device), device_ids=args.gpu_devices, tgt_seq=tgt_seq, mask=tgt_mask)*microbatch/args.batchsize
                        loss.backward()  # data parallism loss is a vector
                        total_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    opt.step()
                    scheduler.step()
                    dset.set_description('Loss: %.4f' % total_loss)
                    if args.wandb:
                        wandb.log({'train/loss': total_loss})
                if (i+1+len(dataloader)*e) % args.sample_freq == 0:
                    bleu_score, edit_distance, token_accuracy = evaluate(model, valdataloader, args, num_batches=int(args.valbatches*e/args.epochs), name='val')
                    save_models(e, step=i, bleu_score=bleu_score,edit_distance=edit_distance,token_accuracy=token_accuracy)
            if (e+1) % args.save_freq == 0:
                save_models(e, step=len(dataloader))
            if args.wandb:
                wandb.log({'train/epoch': e+1})
    except KeyboardInterrupt:
        if e >= 2:
            save_models(e, step=i)
        raise KeyboardInterrupt
    save_models(e, step=len(dataloader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('--resume', help='path to checkpoint folder', action='store_true')
    parsed_args = parser.parse_args()
    if parsed_args.config is None:
        with in_model_path():
            parsed_args.config = os.path.realpath('settings/debug.yaml')
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params), **vars(parsed_args))
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed)
    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()
        wandb.init(config=dict(args), resume='allow', name=args.name, id=args.id)
        args = Munch(wandb.config)
    train(args)
