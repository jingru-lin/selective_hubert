# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import itertools
import logging
import os
import re
import sys
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
import soundfile as sf
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset

import pickle as pkl
import librosa

logger = logging.getLogger(__name__)


def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    to_skip = []
    sids = []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
                to_skip.append(ind)
            elif max_keep is not None and sz > max_keep:
                n_long += 1
                to_skip.append(ind)
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
                speaker = items[0].split("/")[-1].split("-")[0]
                sids.append(speaker)
    tot = ind + 1
    spk2indexes = defaultdict(list)
    for i, sid in enumerate(sids):
        spk2indexes[sid].append(i)

    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    logger.info(
        f"Please check the audio_names and sid match. Audio names = {names[500:550]}"
    )
    logger.info(
        f"Sids = {sids[500:550]}"
    )
    return root, names, dict(spk2indexes), sids, inds, tot, sizes


def load_noise_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, (ind, line, root)
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, loaded {len(names)} "
            f"noise audios, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, tot, sizes


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class SHubertDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        manifest_noise: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
        # speaker
        spk_embedding_dir: str = "",
        # mixing arguments
        get_multiple_prob=0.95,
        noise_apply_prob: float = 0.1,
        noise_db_range: str = "-5_20",
        interf_speech_db_range: str = "-5_5",
        contrastive_data: bool = False,
    ):
        audio_root, audio_names, spk2indexes, sids, inds, tot, sizes = load_audio(
            manifest_path,
            max_keep_sample_size,
            min_keep_sample_size,
        )
        self.audio_root = audio_root
        self.audio_names = audio_names
        self.audio_spk2indexes = spk2indexes
        self.audio_sids = sids
        self.sizes = sizes

        noise_root, noise_names, tot_noise, noise_sizes = load_noise_audio(
            manifest_noise, max_keep_sample_size, min_keep_sample_size
        )
        self.noise_root = noise_root
        self.noise_names = noise_names
        self.noise_sizes = noise_sizes

        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
            )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"len(audio_spk2indexes)={len(self.audio_spk2indexes)}"
        )

        self.noise_apply_prob = noise_apply_prob
        self.get_multiple_prob = get_multiple_prob
        sps = noise_db_range.strip().split("_")
        if len(sps) == 1:
            self.noise_db_low = self.noise_db_high = float(sps[0])
        elif len(sps) == 2:
            self.noise_db_low, self.noise_db_high = float(sps[0]), float(sps[1])
        else:
            raise ValueError("Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]")
        sps = interf_speech_db_range.strip().split("_")
        if len(sps) == 1:
            self.interf_speech_db_low = self.interf_speech_db_high = float(sps[0])
        elif len(sps) == 2:
            self.interf_speech_db_low = float(sps[0])
            self.interf_speech_db_high = float(sps[1])
        else:
            raise ValueError(
                "Format error: '{interf_speech_db_range}' e.g. -3_4 -> [-3db,4db]"
            )
        self.spk_dir = spk_embedding_dir
        self.contrastive_data = contrastive_data

    def get_audio(self, index):
        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_noise_audio(self, wav_length):
        noise_idx = np.random.randint(0, len(self.noise_names))
        noise_path = os.path.join(self.noise_root, self.noise_names[noise_idx])
        wav, cur_sample_rate = sf.read(noise_path)
        if cur_sample_rate != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=cur_sample_rate, target_sr=self.sample_rate)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, self.sample_rate)

        # DNS noise audios are around 10seconds, which is shorter than many LS audios
        # Hence, we sample 1 more noise audio
        # Also, a ratio is applied to allow some clean speech,
        # so that the model can still be trained on some clean speech (also for contrastive loss later)
        if self.get_multiple_prob > 0 and wav_length > len(wav):
            noise_idx = np.random.randint(0, len(self.noise_names))
            noise_path = os.path.join(self.noise_root, self.noise_names[noise_idx])
            wav2, cur_sample_rate = sf.read(noise_path)
            if cur_sample_rate != self.sample_rate:
                wav2 = librosa.resample(wav2, orig_sr=cur_sample_rate, target_sr=self.sample_rate)
            wav2 = torch.from_numpy(wav2).float()
            wav = torch.cat([wav, wav2])
        return wav

    def adjust_noise_energy(self, wav, noise, target_snr):
        power = (wav**2).mean()
        noise_power = (noise**2).mean()
        scale = (
            10 ** (-target_snr / 20) * np.sqrt(power) / np.sqrt(max(noise_power, 1e-10))
        )
        return noise * scale

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        if self.contrastive_data:
            return self._get_parallel(index)
        else:
            return self._get_single(index)

    def load_enroll(self, idx):
        enroll_name = self.audio_names[idx].split('/')[-1]
        enroll_name = enroll_name.split('.')[0]
        # enroll = pkl.load(open(os.path.join(self.spk_dir, enroll_name), 'rb'))
        enroll = np.load(os.path.join(self.spk_dir, enroll_name + '.npy'))
        enroll = torch.from_numpy(enroll)
        return enroll

    def _get_single(self, index):
        wav = self.get_audio(index)
        # choose enrollment
        idx = np.random.choice(self.audio_spk2indexes[self.audio_sids[index]])
        while idx == index and len(self.audio_spk2indexes[self.audio_sids[index]]) > 1:
            idx = np.random.choice(self.audio_spk2indexes[self.audio_sids[index]])
        enroll = self.load_enroll(idx)

        if self.noise_apply_prob >= np.random.random():
            # speech + noise
            noise = self.get_noise_audio(len(wav))
            noise_db = np.random.uniform(self.noise_db_low, self.noise_db_high)
            wav = self.mix_noise(wav, noise, noise_db)
            ratio = 0.0
        else:
            # speech + speech
            index_interf = np.random.randint(0, len(self.audio_names))
            while self.audio_sids[index_interf] == self.audio_sids[index]:
                index_interf = np.random.randint(0, len(self.audio_names))
            interf_speech = self.get_audio(index_interf)
            interf_db = np.random.uniform(
                self.interf_speech_db_low, self.interf_speech_db_high
            )
            wav, ratio = self.mix_audios(wav, interf_speech, interf_db)
            if self.noise_apply_prob >= np.random.random():
                # speech + speech + noise
                noise = self.get_noise_audio(len(wav))
                noise_db = np.random.uniform(self.noise_db_low, self.noise_db_high)
                wav = self.mix_noise(wav, noise, noise_db)

        labels = self.get_labels(index)
        # return {"id": index, "source": wav, "paired_source": None, "enrollment": enroll, "label_list": labels}
        return {"id": index, "source": wav, "enrollment": enroll, "label_list": labels, "ratio": ratio}

    def _get_parallel(self, index):
        wav = self.get_audio(index)
        # clean_wav = wav.clone()
        # choose enrollment
        idx = np.random.choice(self.audio_spk2indexes[self.audio_sids[index]])
        while idx == index and len(self.audio_spk2indexes[self.audio_sids[index]]) > 1:
            idx = np.random.choice(self.audio_spk2indexes[self.audio_sids[index]])
        enroll = self.load_enroll(idx)

        # choose noise
        if self.noise_apply_prob >= np.random.random():
            # if apply noise:
            # Case 1: (s1 + n1) vs (s1 + n2)
            # Case 2: (s1 + n1) vs (s1 + s2)
            # noise_1 = self.get_noise_audio(np.random.randint(0, len(self.noise_names)))
            noise_1 = self.get_noise_audio(len(wav))
            noise_db_1 = np.random.uniform(self.noise_db_low, self.noise_db_high)
            # s1 + n1
            paired_wav = wav.clone()
            wav = self.mix_noise(wav, noise_1, noise_db_1)
            if np.random.random() >= 0.5:
                # s1 + n2
                noise_2 = self.get_noise_audio(len(wav))
                noise_db_2 = np.random.uniform(self.noise_db_low, self.noise_db_high)
                paired_wav = self.mix_noise(paired_wav, noise_2, noise_db_2)
                ratio = 0 # set ratio for noisy speech to 0

            else:
                # s1 + s2
                index_interf_1 = np.random.randint(0, len(self.audio_names))
                while self.audio_sids[index_interf_1] == self.audio_sids[index]:
                    index_interf_1 = np.random.randint(0, len(self.audio_names))
                interf_speech_1 = self.get_audio(index_interf_1)
                interf_db_1 = np.random.uniform(
                    self.interf_speech_db_low, self.interf_speech_db_high
                )
                paired_wav, ratio = self.mix_audios(paired_wav, interf_speech_1, interf_db_1)

        else:
            # if apply interference speech
            # Case 3: (s1 + s2) vs (s1 + s3)
            # Case 4: (s1 + s2) vs (s1 + s2 + n1)
            index_interf_1 = np.random.randint(0, len(self.audio_names))
            while self.audio_sids[index_interf_1] == self.audio_sids[index]:
                index_interf_1 = np.random.randint(0, len(self.audio_names))
            interf_speech_1 = self.get_audio(index_interf_1)
            interf_db_1 = np.random.uniform(
                self.interf_speech_db_low, self.interf_speech_db_high
            )
            if np.random.random() >= 0.5:
                paired_wav = wav.clone()
                # s1 + s2
                wav, ratio = self.mix_audios(wav, interf_speech_1, interf_db_1)
                # s1 + s3 (s3 != s2)
                index_interf_2 = np.random.randint(0, len(self.audio_names))
                while self.audio_sids[index_interf_2] == self.audio_sids[index] or self.audio_sids[index_interf_1] == self.audio_sids[index_interf_2]:
                    index_interf_2 = np.random.randint(0, len(self.audio_names))
                interf_speech_2 = self.get_audio(index_interf_2)
                interf_db_2 = np.random.uniform(
                    self.interf_speech_db_low, self.interf_speech_db_high
                )
                paired_wav, ratio2 = self.mix_audios(paired_wav, interf_speech_2, interf_db_2)
                ratio = max(ratio, ratio2)

            else:
                # s1 + s2
                wav, ratio = self.mix_audios(wav, interf_speech_1, interf_db_1)

                # s1 + s2 + n1
                paired_wav = wav.clone()
                # noise = self.get_noise_audio(np.random.randint(0, len(self.noise_names)))
                noise = self.get_noise_audio(len(wav))
                noise_db = np.random.uniform(self.noise_db_low, self.noise_db_high)
                paired_wav = self.mix_noise(paired_wav, noise, noise_db)

        labels = self.get_labels(index)
        # PLEASE remember to change size accordingly
        return {"id": index, "source": wav, "paired_source": paired_wav, "enrollment": enroll, "label_list": labels, "ratio": ratio}
        # return {"id": index, "source": wav, "paired_source": paired_wav, "clean_source": clean_wav, "enrollment": enroll, "label_list": labels}

    def mix_noise(self, aud, noise, noise_db):
        """
        Noise overlapping regions should be highly overlap
        """
        L = min(len(aud), len(noise))
        aud_start = np.random.randint(0, len(aud) - L + 1)
        noise_start = np.random.randint(0, len(noise) - L + 1)
        # adjust energy
        noise = self.adjust_noise_energy(aud, noise, noise_db)
        aud[aud_start : aud_start + L] += noise[noise_start : noise_start + L]
        return aud

    def mix_audios(self, aud1, aud2, aud2_db):
        """
        Mixture overlapping regions ranges from 0 - 100%
        """
        # adjust length and overlap position
        # use highly overlap for contrastive learning
        #if self.contrastive_data:
        #    L = min(len(aud1), len(aud2))
        #else:
        #    L = min(np.random.randint(1, len(aud1)), len(aud2))
        L = min(np.random.randint(1, len(aud1)), len(aud2))
        # L = min(len(aud1), len(aud2))
        aud1_start = np.random.randint(0, len(aud1) - L + 1)
        aud2_start = np.random.randint(0, len(aud2) - L + 1)
        # adjust energy
        aud2 = self.adjust_noise_energy(aud1, aud2, aud2_db)
        aud1[aud1_start : aud1_start + L] += aud2[aud2_start : aud2_start + L]

        mix_ratio = L / len(aud1)
        return aud1, mix_ratio

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size, audio_start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if audio_start is None:
            if self.random_crop: # True
                start = np.random.randint(0, diff + 1)
        else: # given audio_start
            start = audio_start
        end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        highly_overlapped = max([s["ratio"] for s in samples])
        highly_overlapped = highly_overlapped > 0.5
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        # collate paired source, make sure the paired wavs have the same starts
        if self.contrastive_data:
        # if samples[0]["paired_source"] is not None:
            paired_audios = [s["paired_source"] for s in samples]
            paired_audios_sizes = [len(s) for s in paired_audios]
            assert audio_sizes == paired_audios_sizes
            collated_paired_audios = self.collater_paired_audio(
                paired_audios, audio_size, audio_starts
            )
            # clean_audios = [s["clean_source"] for s in samples]
            # collated_clean_audios = self.collater_paired_audio(
            #     clean_audios, audio_size, audio_starts
            # )

        if highly_overlapped:
            enrolls = [s["enrollment"] for s in samples]
            collated_enrolls = torch.cat(enrolls, axis=0)
        else:
            collated_enrolls = None

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        net_input = {
            "source": collated_audios,
            "padding_mask": padding_mask,
            "spk_emb": collated_enrolls,
        }
        if self.contrastive_data:
        # if samples[0]["paired_source"] is not None:
            net_input["paired_source"] = collated_paired_audios
            # net_input["clean_source"] = collated_clean_audios
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts


    def collater_paired_audio(self, audios, audio_size, audio_starts):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
            else:
                collated_audios[i], _ = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i]
                )
        return collated_audios

    def collater_frm_label(self, targets, audio_size, audio_starts, label_rate, pad):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        if self.contrastive_data:
            return min(self.sizes[index], self.max_sample_size) * 2
        else:
            return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
