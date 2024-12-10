import csv
import json
import math
import os
import random
import sys

import h5py
import numpy as np
import soundfile as sf
from scipy import signal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DistributedSampler, WeightedRandomSampler

import itertools

class DistributedSamplerWrapper(DistributedSampler):
    # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238

    def __init__(
        self, sampler, dataset,
        num_replicas=None,
        rank=None,
        shuffle: bool=True
    ):
        super(DistributedSamplerWrapper, self).__init__(dataset, num_replicas, rank, shuffle)
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
        

class DistributedWeightedSampler(Sampler):
    # dataset_train, samples_weight,  num_replicas=num_tasks, rank=global_rank
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.weights = torch.from_numpy(weights)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # targets = self.dataset.targets
        # # select only the wanted targets for this subsample
        # targets = torch.tensor(targets)[indices]
        # assert len(targets) == self.num_samples
        # # randomly sample this subset, producing balanced classes
        # weights = self.calculate_weights(targets)
        weights = self.weights[indices]

        subsample_balanced_indicies = torch.multinomial(weights, self.num_samples, self.replacement)
        # now map these target indicies back to the original dataset index...
        dataset_indices = torch.tensor(indices)[subsample_balanced_indicies]
        return iter(dataset_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        line_count = 0
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            print(row)
            index, mid = row[0], row[1]
            display_name = ",".join(row[2:])
            # index_lookup[row['mid']] = row['index']
            #index_lookup[row['mid']] = line_count
            index_lookup[display_name] = line_count
            line_count += 1
    return index_lookup

def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = np.sqrt(np.mean(audio_data**2)) # Calculate the RMS of the audio
   
    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data
    
    current_dBFS = 20 * np.log10(rms) # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20) # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear # Apply the gain to the audio data
    return normalized_audio

class MultichannelDataset(Dataset):
    def __init__(
            self, 
            audio_json, audio_conf, audio_path_root, meta_path_root,
            reverb_json, reverb_type, reverb_path_root,
            label_csv=None, roll_mag_aug=False, normalize=True,
            _ext_audio=".wav", mode="train"
        ):

        self.data = json.load(open(audio_json, 'r'))['data']
        self.audio_path_root = audio_path_root
        self.meta_path_root = meta_path_root

        self.reverb_path_root = reverb_path_root
        self.reverb = json.load(open(reverb_json, 'r'))['data']
        self.reverb_type = reverb_type
        self.channel_num = 2 if reverb_type == 'binaural' else 9 if reverb_type == 'ambisonics' else 4 if reverb_type == "tetra" else 1
        
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        if 'multilabel' in self.audio_conf.keys():
            self.multilabel = self.audio_conf['multilabel']
        else:
            self.multilabel = False
        self.mixup = self.audio_conf.get('mixup')
        self.dataset = self.audio_conf.get('dataset')
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        
        self.roll_mag_aug = roll_mag_aug
        self.normalize = normalize
        
        self._ext_audio = _ext_audio
        self.mode = mode

        print(f'multilabel: {self.multilabel}')
        print(f'using mix-up with rate {self.mixup}')
        print(f'number of classes: {self.label_num}')
        print(f'size of dataset: {self.__len__()}')

    def _roll_mag_aug(self, waveform):
        idx = np.random.randint(len(waveform))
        mag = np.random.beta(10, 10) + 0.5
        return torch.roll(waveform, idx) * mag

    def fetch_spatial_targets(self, reverb_item):
        sensor_position = np.array([float(i) for i in reverb_item['sensor_position'].split(',')])
        source_position = np.array([float(i) for i in reverb_item['source_position'].split(',')])
        distance = np.linalg.norm(sensor_position - source_position)
        distance = round(distance * 2) # 21 classes
        
        # NOTE: pay attention to the coordinate system
        dx = source_position[0] - sensor_position[0] # LEFT-RIGHT
        dy = source_position[1] - sensor_position[1] # UP-DOWN
        dz = source_position[2] - sensor_position[2] # FRONT-BACK

        azimuth_degrees = math.degrees(math.atan2(-dz, dx)) # degree
        azimuth_degrees = (round(azimuth_degrees) + 360) % 360 # [-180, -0] -- > [+180, +360]; [0, 180] --> [0, +180]
        elevation_degrees = math.degrees(math.atan(dy / math.sqrt(dx**2 + dz**2))) # degree
        elevation_degrees = (round(elevation_degrees) + 90) % 180 # [-90, 90] --> [0, 179], need reverse

        spaital_targets = { 
            "distance": distance,         
            "azimuth": azimuth_degrees,
            "elevation": elevation_degrees        
        }   
        return spaital_targets


    def fetch_rounded_spatial_targets(self, targets):
        
        spaital_targets = { 
            "distance": [round(dist * 2) for dist in targets["distance"]],         
            "azimuth": [(round(azi)+ 360) % 360 for azi in targets["azimuth"]],
            "elevation": [(round(ele) + 90) % 180 for ele in targets["elevation"]]       
        } 
        return spaital_targets
    
    def __getitem__(self, index):
        """
        Fetches and processes a single audio sample and its associated data.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: Processed waveform, reverb data, label indices, spatial targets, audio path, reverb path.
        """
        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        audio_path = os.path.join(self.audio_path_root, datum['fname'] + self._ext_audio)

        adpit_path = os.path.join(self.meta_path_root, datum['fname'] + ".npy")
        adpit_label = np.load(adpit_path)
        adpit_label = torch.from_numpy(adpit_label).float()

        spaital_targets = self.fetch_rounded_spatial_targets(datum)
        
        waveform, sr = sf.read(audio_path)
        if waveform.shape[1] == 32: # leave this for now until we get the 32 channel fixed to 4ch only
            waveform = waveform[:, [5, 9, 25, 21]]
        waveform = signal.resample_poly(waveform, 32000, sr) if sr != 32000 else waveform   
        waveform = torch.from_numpy(waveform.T).float()

        mix_lambda = np.random.beta(10, 10)
        label_indices = np.zeros(self.label_num * 2)  # initialize the label (*2 since we predict 2 classes now)

        for i, label_str in enumerate(datum['class']):
            label_indices[self.index_dict[label_str] + i*self.label_num] = 1.0

        label_indices = torch.FloatTensor(label_indices)
        return waveform, None, label_indices, spaital_targets, audio_path, None, adpit_label

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        waveforms, reverbs, label_indices, spatial_targets, audio_path, reverb_path, adpit_label = zip(*batch)
        waveforms = torch.stack(waveforms)

        adpit_label = torch.stack(adpit_label)
        B = waveforms.shape[0]
        
        # spaital_targets
        spatial_targets_dict = {
            'distance': torch.zeros((B, 2), dtype=torch.long),
            'azimuth': torch.zeros((B, 2), dtype=torch.long),
            'elevation': torch.zeros((B, 2), dtype=torch.long)
        }

        # Process spatial targets
        for i, target in enumerate(spatial_targets):
            spatial_targets_dict['distance'][i] = torch.cat([
                torch.tensor(target['distance'], dtype=torch.long), 
                torch.zeros(2 - len(target['distance']), dtype=torch.long)
            ])
            spatial_targets_dict['azimuth'][i] = torch.cat([
                torch.tensor(target['azimuth'], dtype=torch.long), 
                torch.zeros(2 - len(target['azimuth']), dtype=torch.long)
            ])
            spatial_targets_dict['elevation'][i] = torch.cat([
                torch.tensor(target['elevation'], dtype=torch.long), 
                torch.zeros(2 - len(target['elevation']), dtype=torch.long)
            ])

        return waveforms, None, torch.stack(label_indices), spatial_targets_dict, audio_path, None, adpit_label
