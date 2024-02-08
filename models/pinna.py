# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from lava.lib.dl import slayer

class Network(torch.nn.Module):
    def __init__(self,
            threshold=0.1, 
            tau_grad=0.1, 
            scale_grad=0.8,
            max_delay=64, 
            out_delay=0,
            n_fft=257,
            use_phase_input=False,
            produce_phase_output=False,
            input_is_binaural=False,
            output_is_binaural=False):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay

        sigma_params = { # sigma-delta neuron parameters
            'threshold'     : threshold,   # delta unit threshold
            'tau_grad'      : tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,  # trainable threshold
            'shared_param'  : True,   # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : F.relu, # activation function
        }

        self.input_quantizer = lambda x: slayer.utils.quantize(x, step=1 / 64)

        input_width = n_fft
        if (input_is_binaural):
            input_width *= 2
        if (use_phase_input):
            input_width *= 2

        output_width = n_fft
        if (output_is_binaural):
            putput_width *= 2
        if (produce_phase_output):
            output_width *= 2

        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Dense(sdnn_params, input_width, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Dense(sdnn_params, 512, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Output(sdnn_params, 512, output_width, weight_norm=False),
        ])

        self.blocks[0].pre_hook_fx = self.input_quantizer

        self.blocks[1].delay.max_delay = max_delay
        self.blocks[2].delay.max_delay = max_delay

    def _segmental_snr_mixer(self, clean, noise, snr,
                        target_level,
                        target_level_lower,
                        target_level_higher,
                        clipping_threshold=0.99,
                        ):
        '''Function to mix clean speech and noise at various segmental SNR levels'''
        clean_div = torch.max(torch.abs(clean)) + self.EPS
        noise_div = torch.max(torch.abs(noise)) + self.EPS
        ssl_clean = torch.div(clean, clean_div.item())
        ssl_noise = torch.div(noise, noise_div.item())
        # TODO should only calculate the RMS of the 'active' windows, but
        # for now we just use the whole audio sample
        clean_rms = torch.sqrt(torch.mean(torch.square(ssl_clean))).item()
        noise_rms = torch.sqrt(torch.mean(torch.square(ssl_noise))).item()
        clean_scalar = 10 ** (target_level / 20) / (clean_rms + self.EPS)
        noise_scalar = 10 ** (target_level / 20) / (noise_rms + self.EPS)
        ssl_clean = torch.mul(ssl_clean, clean_scalar)
        ssl_noise = torch.mul(ssl_noise, noise_scalar)
        # Adjust noise to SNR level
        noise_scalar = clean_rms / (10**(snr/20)) / (noise_rms+self.EPS)
        ssl_noise = torch.mul(ssl_noise, noise_scalar)
        ssl_noisy = torch.add(ssl_clean, ssl_noise)
        noisy_rms_level = torch.randint(
                target_level_lower,
                target_level_higher,
                (1,))
        noisy_rmsT = torch.sqrt(torch.mean(torch.square(ssl_noisy)))
        noisy_rms = torch.sqrt(torch.mean(torch.square(ssl_noisy))).item()
        noisy_scalar = 10 ** (noisy_rms_level / 20) / (noisy_rms + self.EPS)
        ssl_noisy = torch.mul(ssl_noisy, noisy_scalar.item())
        ssl_clean = torch.mul(ssl_clean, noisy_scalar.item())
        ssl_noise = torch.mul(ssl_noise, noisy_scalar.item())
        # check if any clipping happened
        needToClip = torch.gt(torch.abs(ssl_noisy), 0.99).any() # 0.99 is the clipping threshold
        if (needToClip):
            noisyspeech_maxamplevel = torch.max(torch.abs(ssl_noisy)).item() / (0.99 - self.EPS)
            ssl_noisy = torch.div(ssl_noisy, noisyspeech_maxamplevel)
            ssl_noise = torch.div(ssl_noise, noisyspeech_maxamplevel)
            ssl_clean = torch.div(ssl_clean, noisyspeech_maxamplevel)
            noisy_rms_level = int(20 * np.log10(noisy_scalar/noisyspeech_maxamplevel * (noisy_rms + self.EPS)))
        return ssl_clean, ssl_noise, ssl_noisy, noisy_rms_level

    def synthesizeNoisySpeech(self, clean, noise, batchSize,
            snr,
            targetLevel,
            targetLevelLower,
            targetLevelHigher):
        ssl_noisy = torch.zeros(batchSize, 480000).to(device)
        ssl_noise = torch.zeros(batchSize, 480000).to(device)
        ssl_clean = torch.zeros(batchSize, 480000).to(device)
        for i in range(batchSize):
            ssl_clean[i, :], ssl_noise[i,:], ssl_noisy[i,:], rms = self._segmental_snr_mixer(clean[i,:], noise[i,:],
                snr[i].item(),
                targetLevel[i].item(),
                targetLevelLower,
                targetLevelHigher)

        return ssl_noisy, ssl_clean, ssl_noise

    def forward(self, noisy):
        x = noisy - self.stft_mean

        for block in self.blocks:
            x = block(x)

        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def validate_gradients(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any()
                                       or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            self.zero_grad()

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

        
