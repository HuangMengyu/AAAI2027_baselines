import numpy as np
import torch
import random


# def DataTransform(sample, config):

#     weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
#     strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

#     return weak_aug, strong_aug

def DataTransform_time_freq(time, freq):
    # batch_size, channel, timestep = time.shape
    # time = np.array(time)
    # freq = np.array(freq)

    # augmentation for time
    time_augs = [scaling, jitter, masking]
    chosen_time_aug = random.choice(time_augs)
    time_aug_result = chosen_time_aug(time)

    # augmentation for freq
    freq_augs = [remove_frequency, add_frequency]
    chosen_freq_aug = random.choice(freq_augs)
    freq_aug_result = chosen_time_aug(freq)

    return time_aug_result, freq_aug_result

def DataTransform_time(time):
    # batch_size, channel, timestep = time.shape
    # time = np.array(time)
    # freq = np.array(freq)

    # augmentation for time
    time_augs = [scaling, jitter, masking]
    chosen_time_aug = random.choice(time_augs)
    time_aug_result = chosen_time_aug(time)

    return time_aug_result



def jitter(x, sigma=2): #for time input
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.5): #for tiem input
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

def generate_binomial_mask(B, T, D, p=0.5): # p is the ratio of not zero
    # return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)
    return np.random.binomial(1, p, size=(B, T, D)).astype(bool)

def masking(x, drop_ratio=0.2, mask= 'binomial'):
    global mask_id
    # nan_mask = ~x.isnan().any(axis=-1)
    nan_mask = ~np.isnan(x).any(axis=-1)
    x[~nan_mask] = 0
    # x = self.input_fc(x)  # B x T x Ch

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.shape[0], x.shape[1], x.shape[2], p=1-drop_ratio)
    # elif mask == 'continuous':
    #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    # elif mask == 'all_true':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    # elif mask == 'all_false':
    #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    # elif mask == 'mask_last':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     mask[:, -1] = False

    # mask &= nan_mask
    x[~mask_id] = 0
    return x



def remove_frequency(x, pertub_ratio=0.1): # x is the magnutude of complex value fft input
    # out = x.abs()
    mask = np.random.uniform(size=out.shape) > pertub_ratio # maskout_ratio are False
    mask = mask
    return out*mask

def add_frequency(x, pertub_ratio=0.1): # x is the magnitude of complex value fft input
    # out = x.abs()
    mask = np.random.uniform(size=out.shape) > (1 - pertub_ratio) # only pertub_ratio of all values are True
    mask = mask
    max_amplitude = out.max()
    random_am = np.random.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return out+pertub_matrix

# not sure how the phase permutation is implemented
# def phase_permutation(x, pertub_ratio=0.1, seg_mode="random"): # x is complex value fft input
#     # compute phase and magnitude first
#     a = freq.real
#     b = freq.imag

#     magnitude = np.abs(freq)

#     phase = np.zeros_like(a)
#     phase[a > 0] = np.arctan(b / a)[a > 0]
#     phase[a < 0] = (np.arctan(b / a) + np.sign(b) * np.pi)[a < 0]
#     phase[a == 0] = (np.sign(b) * np.pi / 2)[a == 0]


#     orig_steps = np.arange(phase.shape[2])

#     max_segments = int(phase.shape[2]*pertub_ratio)
#     num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

#     ret = np.zeros_like(phase)
#     for i, pat in enumerate(phase):
#         if num_segs[i] > 1:
#             if seg_mode == "random":
#                 split_points = np.random.choice(phase.shape[2] - 2, num_segs[i] - 1, replace=False)
#                 split_points.sort()
#                 splits = np.split(orig_steps, split_points)
#             else:
#                 splits = np.array_split(orig_steps, num_segs[i])
#             warp = np.concatenate(np.random.permutation(splits)).ravel()
#             ret[i] = pat[0,warp]
#         else:
#             ret[i] = pat
    

#     return torch.from_numpy(ret)

