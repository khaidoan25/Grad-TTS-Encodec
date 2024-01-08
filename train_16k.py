# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params_16k
from model import GradTTSExt
# from data import TextMelDataset, TextMelBatchCollate
from data import TextAudioCodeDataset, TextAudioCodeBatchCollate
from utils import plot_tensor, save_plot
import utils
from text.symbols import symbols
from data_utils.audio_processor import AudioTokenizerExt

import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram

from accelerate import Accelerator

# train_filelist_path = params_16k.train_filelist_path
# valid_filelist_path = params_16k.valid_filelist_path
train_cuts = params_16k.train_cuts_path
valid_cuts = params_16k.valid_cuts_path
cmudict_path = params_16k.cmudict_path
add_blank = params_16k.add_blank

log_dir = params_16k.log_dir
n_epochs = params_16k.n_epochs
batch_size = params_16k.batch_size
out_size = params_16k.out_size
learning_rate = params_16k.learning_rate
random_seed = params_16k.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params_16k.n_enc_channels
filter_channels = params_16k.filter_channels
filter_channels_dp = params_16k.filter_channels_dp
n_enc_layers = params_16k.n_enc_layers
enc_kernel = params_16k.enc_kernel
enc_dropout = params_16k.enc_dropout
n_heads = params_16k.n_heads
window_size = params_16k.window_size

n_feats = params_16k.n_feats
n_fft = params_16k.n_fft
n_mels = params_16k.n_mels
sample_rate = params_16k.sample_rate
hop_length = params_16k.hop_length
win_length = params_16k.win_length
f_min = params_16k.f_min
f_max = params_16k.f_max

dec_dim = params_16k.dec_dim
beta_min = params_16k.beta_min
beta_max = params_16k.beta_max
pe_scale = params_16k.pe_scale

tokenizer = AudioTokenizerExt(sample_rate=sample_rate, device="cpu")
global_step = params_16k.global_step

if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    accelerator = Accelerator()

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    train_dataset = TextAudioCodeDataset(train_cuts, cmudict_path, add_blank)
    batch_collate = TextAudioCodeBatchCollate()
    train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          collate_fn=batch_collate, drop_last=True,
                          num_workers=4, shuffle=True)
    test_dataset = TextAudioCodeDataset(valid_cuts, cmudict_path, add_blank)

    print('Initializing model...')
    model = GradTTSExt(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    
    try:
        model.cuda()
        model, prev_epoch = utils.load_checkpoint(log_dir, model)
    except:
        print("Training grand new model...!")

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params_16k.test_size)
    for i, item in enumerate(test_batch):
        emb = item['y']
        code = tokenizer.encode_emb(torch.Tensor(emb).unsqueeze(0))
        samples = tokenizer.decode(code).squeeze().unsqueeze(0)
        mel = mel_spectrogram(samples, n_fft, n_mels, sample_rate, hop_length, win_length, f_min, f_max, center=False)
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')
        
    train_dl, model, optimizer = accelerator.prepare(
        train_dl, model, optimizer
    )
    world_size = accelerator.state.num_processes

    print('Start training...')
    iteration = global_step
    for epoch in range(1, n_epochs + 1):
        epoch += prev_epoch
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(train_dl, total=len(train_dataset)//(batch_size*world_size)) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                dur_loss, prior_loss, diff_loss = model(x, x_lengths,
                                                        y, y_lengths,
                                                        out_size=out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                # loss.backward()
                accelerator.backward(loss)

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                logger.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                
                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                    progress_bar.set_description(msg)
                
                iteration += 1

        log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % params_16k.save_every > 0:
            continue

        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                y_enc, y_dec, attn = model.module.gen(x, x_lengths, n_timesteps=50)
                # Y_t
                code_enc = tokenizer.encode_emb(y_enc.cpu())
                samples_enc = tokenizer.decode(code_enc).squeeze().unsqueeze(0)
                mel_enc = mel_spectrogram(samples_enc, n_fft, n_mels, sample_rate, hop_length, win_length, f_min, f_max, center=False)
                # Y_0
                code_dec = tokenizer.encode_emb(y_dec.cpu())
                samples_dec = tokenizer.decode(code_dec).squeeze().unsqueeze(0)
                mel_dec = mel_spectrogram(samples_dec, n_fft, n_mels, sample_rate, hop_length, win_length, f_min, f_max, center=False)
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(mel_enc.squeeze()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(mel_dec.squeeze()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(), 
                          f'{log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(), 
                          f'{log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(), 
                          f'{log_dir}/alignment_{i}.png')

        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
