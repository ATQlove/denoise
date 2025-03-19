'''Main file for running experiments.'''

import time
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from filters.wavlet_filter import apgd, bsplinelinear
from filters.quadratic_filter import quadratic_filter
from filters.TV_filter import TV_filter
from filters.TV_filter_pd import TV_filter_pd
from filters.tikhonov_filter import tikhonov
from filters.non_local_means_filter import non_local_means_filter
from filters.non_local_wnnm_filter import non_local_wnnm_filter
from utilities.utils import (
    read_image, add_gaussian_noise, add_poisson_noise,
    PSNR, create_results_directory, normalize_image
)



def main(noise_type='gaussian', plot=False, savefigs=True):
    '''Runs the various filters on the provided images with varying noise levels
       and saves the results.

    Args:
        noise_type (str): The type of noise, either 'gaussian' or 'poisson'.
        plot (bool): Whether to plot the noisy and cleaned images.
        savefigs (bool): Whether to save the generated images.
    '''

    np.random.seed(0)

    PSNR_results = {
        'quad': {}, 'TV': {}, 'tik': {}, 'nlm': {}, 'wnnm': {}, 'wavelet': {}
    }
    time_results = {
        'quad': {}, 'TV': {}, 'tik': {}, 'nlm': {}, 'wnnm': {}, 'wavelet': {}
    }

    images = ['landscape_gray', 'peacock_gray', 'spray_gray']

    if noise_type == 'gaussian':
        hyperparameters = [0.01, 0.025, 0.05]
    elif noise_type == 'poisson':
        hyperparameters = [50, 20, 10]

    # # For test
    # images = ['clock']
    # if noise_type == 'gaussian':
    #     hyperparameters = [0.01]
    # elif noise_type == 'poisson':
    #     hyperparameters = [50]


    create_results_directory(noise_type, images, hyperparameters)

    for im_name in images:
        for key in PSNR_results.keys():
            PSNR_results[key][im_name] = []
            time_results[key][im_name] = []

        for param in hyperparameters:
            str_var = str(param).replace('.', '_')

            im = read_image(im_name)
            if noise_type == 'gaussian':
                noisy_im = add_gaussian_noise(im, mean=0, var=param)
                variance = param
            elif noise_type == 'poisson':
                noisy_im = add_poisson_noise(im, photons=param)
                variance = 0.5/param

            start_time = time.time()

            # ============= Quadratic Filter =============
            quad_im = quadratic_filter(noisy_im, 5)
            quad_time = time.time()
            quad_im = normalize_image(quad_im)

            # ============= TV Filter =============
            # TV_im = TV_filter(noisy_im, 0.3)
            TV_im = TV_filter_pd(noisy_im, 6)
            TV_time = time.time()
            TV_im = normalize_image(TV_im)

            # ============= Tikhonov Filter =============
            tik_start = time.time()
            # change numpy to torch, then use tikhonov
            tik_tensor = tikhonov(torch.from_numpy(noisy_im))
            # return is torch.Tensor, so change it back to numpy
            tik_im = tik_tensor.detach().cpu().numpy()
            tik_im = normalize_image(tik_im)
            tik_end = time.time()

            # ============= Non-local Means =============
            nlm_im = non_local_means_filter(noisy_im, 7, 10, 0.1)
            nlm_time = time.time()
            nlm_im = normalize_image(nlm_im)

            # ============= Weighted Nuclear Norm Minimization =============
            x = noisy_im
            y = noisy_im
            delta = 0.3
            for _ in range(1):
                y = x + delta*(noisy_im - y)
                x = non_local_wnnm_filter(y, 7, 10, variance)
                x = normalize_image(x)

            wnnm_im = x

            wnnm_time = time.time()

            # ============= Wavelet Denoising (APGD) =============
            wav_start = time.time()
            
            A = np.ones_like(noisy_im)

            wav_im = apgd(
                f=noisy_im,
                A=A,
                thresh=[0.1, 0.07, 0.04, 0.01],
                masks=bsplinelinear,  # using linear B mask, can change to haar, db2, db3
                levels=4, # 4 layers
                iters=20,
                verbose=False,   # if you want to see process, change to True
                showiters=False  # if you want to see the image in realtime, change it to True
            )
            
            wav_im = normalize_image(wav_im)
            wav_end = time.time()

            # ============= PSNR =============
            PSNR_results['quad'][im_name].append(PSNR(original_im=im, cleaned_im=quad_im))
            PSNR_results['TV'][im_name].append(PSNR(original_im=im, cleaned_im=TV_im))
            PSNR_results['tik'][im_name].append(PSNR(original_im=im, cleaned_im=tik_im))
            PSNR_results['nlm'][im_name].append(PSNR(original_im=im, cleaned_im=nlm_im))
            PSNR_results['wnnm'][im_name].append(PSNR(original_im=im, cleaned_im=wnnm_im))
            PSNR_results['wavelet'][im_name].append(PSNR(original_im=im, cleaned_im=wav_im))

            # ============= time =============
            time_results['quad'][im_name].append(quad_time - start_time)
            time_results['TV'][im_name].append(TV_time - quad_time)
            time_results['tik'][im_name].append(tik_end - tik_start)
            time_results['nlm'][im_name].append(nlm_time - TV_time)
            time_results['wnnm'][im_name].append(wnnm_time - nlm_time)
            time_results['wavelet'][im_name].append(wav_end - wav_start)

            # ============= plot & save =============
            if plot is True or savefigs is True:
                _, ax_original = plt.subplots()
                ax_original.imshow(im, cmap='gray')
                ax_original.set_title('Original Image')

                fig_noisy, ax_noisy = plt.subplots()
                ax_noisy.imshow(noisy_im, cmap='gray')
                ax_noisy.set_title(f'Noisy Image, PSNR={round(PSNR(original_im=im, cleaned_im=noisy_im), 2)}')

                fig_quad, ax_quad = plt.subplots()
                ax_quad.imshow(quad_im, cmap='gray')
                ax_quad.set_title(f'Quadratic Image, PSNR={round(PSNR(original_im=im, cleaned_im=quad_im), 2)}')

                fig_tv, ax_tv = plt.subplots()
                ax_tv.imshow(TV_im, cmap='gray')
                ax_tv.set_title(f'TV Image, PSNR={round(PSNR(original_im=im, cleaned_im=TV_im), 2)}')

                fig_tik, ax_tik = plt.subplots()
                ax_tik.imshow(tik_im, cmap='gray')
                ax_tik.set_title(f'Tikhonov Image, PSNR={round(PSNR(original_im=im, cleaned_im=tik_im), 2)}')

                fig_nlm, ax_nlm = plt.subplots()
                ax_nlm.imshow(nlm_im, cmap='gray')
                ax_nlm.set_title(f'Non-local means Image, PSNR={round(PSNR(original_im=im, cleaned_im=nlm_im), 2)}')

                fig_wnnm, ax_wnnm = plt.subplots()
                ax_wnnm.imshow(wnnm_im, cmap='gray')
                ax_wnnm.set_title(f'Weighted Nuclear Norm Minimization Image, PSNR={round(PSNR(original_im=im, cleaned_im=wnnm_im), 2)}')

                fig_wav, ax_wav = plt.subplots()
                ax_wav.imshow(wav_im, cmap='gray')
                ax_wav.set_title(f'Wavelet Image, PSNR={round(PSNR(original_im=im, cleaned_im=wav_im), 2)}')


                if savefigs is True:
                    fig_noisy.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/noisy.png')
                    fig_quad.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/quad.png')
                    fig_tv.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/tv.png')
                    fig_tik.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/tik.png')
                    fig_nlm.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/nlm.png')
                    fig_wnnm.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/wnnm.png')
                    fig_wav.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/wavelet.png')

                if plot is True:
                    plt.show()

    print(PSNR_results)
    print(time_results)
    with open(f'./results/{noise_type}/PSNR_results.pkl', 'wb') as f:
        pickle.dump(PSNR_results, f)
    with open(f'./results/{noise_type}/time_results.pkl', 'wb') as f:
        pickle.dump(time_results, f)


if __name__ == '__main__':
    main()
