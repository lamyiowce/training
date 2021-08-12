# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import numpy as np
import tensorflow as tf

from experiments.ml.specaugment.mlcommons.training.rnn_speech_recognition.pytorch.common.data.text import Tokenizer
from experiments.ml.specaugment.mlcommons.training.rnn_speech_recognition.pytorch.tf_utils import melscale, dbscale, \
    stack_subsample_frames
from experiments.ml.specaugment.specaugment import spec_augment


def add_periodic_caching(dataset, period, snapshot_path):
    snapshot_path = snapshot_path + "snapshot-{:02d}"
    dataset_snapshot = None
    for ep in range(3):
        d = dataset.apply(tf.data.experimental.snapshot(snapshot_path.format(ep), compression=None))
        d = d.repeat(period)
        if ep == 0:
            dataset_snapshot = d
        else:
            dataset_snapshot = dataset_snapshot.concatenate(d)
    return dataset_snapshot


def do_snapshot(ds, caching_period, snapshot_path):
    if caching_period == -1:
        assert snapshot_path
        ds = ds.apply(tf.data.experimental.snapshot(snapshot_path, compression=None))
    elif caching_period >= 1:
        add_periodic_caching(ds, caching_period, snapshot_path)
    return ds


class TfDataset:
    def __init__(self,
                 pipeline,
                 caching_period,
                 tokenizer: Tokenizer,
                 dataset_path,
                 manifests_paths,
                 batch_size,
                 config_data,
                 config_features,
                 snapshot_path=None,
                 num_elems_to_load=None,
                 service_ip=None,
                 wav=False,
                 repeat_single_batch=False,
                 even_batches=False,
                 drop_remainder=False,
                 sort=False,
                 ):
        self.dataset_path = dataset_path
        self.manifests_paths = manifests_paths
        self.drop_remainder = drop_remainder
        if 'snapshot' in pipeline:
            assert snapshot_path
        self.pipeline = pipeline
        self.caching_period = caching_period
        self.tokenizer = tokenizer
        self.snapshot_path = snapshot_path
        self.repeat_single_batch = repeat_single_batch
        self.num_elems_to_load = num_elems_to_load  # if not repeat_single_batch else 64 # Assuming this will be > than the batch size
        self.service_ip = service_ip
        self.wav = wav
        self.even_batches = even_batches
        self.batch_size = batch_size
        self.sort = sort

        self.max_duration_seconds = config_data['max_duration']

        self.window_size = config_features['window_size']
        self.window_stride = config_features['window_stride']
        self.nfeatures = config_features['n_filt']
        self.nfft = config_features['n_fft']
        self.preemph_coeff = .97

    def __len__(self):
        return len(self.entries)

    def read_entries(self):
        self.entries = []
        logger = tf.get_logger()
        for file_path in self.manifests_paths:
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()[1:]
                limit = self.num_elems_to_load or len(temp_lines)
                # Skip the header of tsv file
                self.entries += temp_lines[:limit]
        # The files is "\t" seperated
        self.entries = [line.split("\t", 2) for line in self.entries
                        if float(line.split("\t", 2)[1]) <= self.max_duration_seconds]
        for i, line in enumerate(self.entries):
            self.entries[i][-1] = " ".join([str(x) for x in self.tokenizer.tokenize(line[-1])])
        self.entries = np.array(self.entries)
        if self.sort:
            durs = self.entries[:, 1].astype('f')
            ord = np.argsort(-durs)
            self.entries = self.entries[ord, :]
        self.total_steps = len(self.entries)

    # @staticmethod
    # def load(record: tf.Tensor):
    #     import tensorflow_io as tfio
    #     path = record[0]
    #     audio = tfio.audio.AudioIOTensor(path, dtype=tf.int16)
    #     audio = tf.cast(audio.to_tensor(), tf.float32) / 32768.0
    #     audio = tf.squeeze(audio, axis=-1)  # Make shape (LEN,)
    #     return audio, record[2]

    @tf.function
    def load_wav(self, record: tf.Tensor):
        path = record[0]
        audio, _rate = tf.audio.decode_wav(tf.io.read_file(path))
        audio = tf.squeeze(audio, axis=-1)  # Make shape (LEN,)
        return audio, record[2]

    @tf.function
    def _preprocess_audio(self, audio: tf.Tensor):
        # Preprocessing hyperparameters from Lingvo:
        # https://github.com/tensorflow/lingvo/blob/master/lingvo/tools/audio_lib.py
        FRAME_STEP_SECONDS = 10. / 1000
        # FRAME_STEP_SECONDS = self.window_stride  # 10. / 1000
        # FRAME_LENGTH_SECONDS = self.window_size  # 25. / 1000
        FRAME_LENGTH_SECONDS = 25. / 1000
        MEL_BINS = self.nfeatures
        MEL_LOWER_BOUND_HERTZ = 125.
        MEL_UPPER_BOUND_HERTZ = 7600.
        RATE = 16000

        if self.preemph_coeff > 0:
            audio = audio[1:] - self.preemph_coeff * audio[:-1]

        # spectrogram using stft
        frame_len = tf.cast(tf.math.round(FRAME_LENGTH_SECONDS * RATE), tf.int32)
        frame_step = tf.cast(tf.math.round(FRAME_STEP_SECONDS * RATE), tf.int32)

        x = tf.abs(tf.signal.stft(audio, frame_length=frame_len, frame_step=frame_step, fft_length=self.nfft))
        x = tf.math.pow(x, 0.5)

        x = melscale(
            x, rate=RATE, mels=MEL_BINS, fmin=MEL_LOWER_BOUND_HERTZ, fmax=MEL_UPPER_BOUND_HERTZ)
        x = dbscale(x, top_db=80)

        # Standardization.
        means = tf.math.reduce_mean(x, 1, keepdims=True)
        stddevs = tf.math.reduce_std(x, 1, keepdims=True)
        x = tf.math.divide_no_nan((x - means), stddevs)

        return x

    def create_inputs(self, audio, indices):
        audio = tf.transpose(audio)

        # Audio shape (80, None)
        audio = stack_subsample_frames(audio, 3, 3)
        # Audio shape (240, None)

        # audio_length = tf.cast(tf.shape(audio)[1], tf.int32)
        audio_length = tf.cast(tf.shape(audio)[1], tf.float32)
        print(audio.shape)
        label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32)
        label_length = tf.cast(tf.shape(label)[0], tf.int32)
        # Inputs, Targets, inputs percentage of max len, target len
        return audio, audio_length, label, label_length

    def pad_audio(self, x, indices):
        length = 2453
        paddings = tf.constant([[0, length], [0, 0], [0, 0]])
        x = tf.pad(x, paddings, "CONSTANT")[:length, :, :]
        x = tf.ensure_shape(x, [length, 80])

        indices_length = 398
        indices_paddings = tf.constant([[0, indices_length]])
        indices = tf.pad(tf.strings.split(indices), indices_paddings, "CONSTANT", constant_values="0")[
                  :indices_length]
        indices = tf.ensure_shape(indices, [indices_length])
        return x, indices

    def create_tf_dataset(self):
        self.read_entries()

        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        if self.wav:
            dataset = dataset.map(self.load_wav, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            raise NotImplementedError
            # dataset = dataset.map(self.load, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda audio, transcript: (self._preprocess_audio(audio), transcript),
                              num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.map(lambda audio, transcript: (audio, self.vectorizer.parse_transcript(transcript)),
        #                       num_parallel_calls=tf.data.AUTOTUNE)
        logger = tf.get_logger()
        # for i, line in enumerate(self.entries):
        #     self.entries[i][-1] = self.vectorizer.parse_transcript(line[-1])

        for stage in self.pipeline:
            if stage == 'augment':
                logger.info("Augmenting...")
                dataset = dataset.map(
                    lambda audio, transcript: (
                        spec_augment(audio, time_warping_para=80, frequency_masking_para=27,
                                     time_masking_para=100),
                        transcript
                    ),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            elif stage == 'snapshot':
                logger.info("Peforming snapshot...")
                dataset = do_snapshot(dataset, self.caching_period, self.snapshot_path)
            else:
                raise ValueError(f'No such pipeline stage: {stage}')

        if self.even_batches:
            dataset = dataset.map(self.pad_audio, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.create_inputs, num_parallel_calls=tf.data.AUTOTUNE)
        if self.even_batches:
            dataset = dataset.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE,
                                    drop_remainder=self.drop_remainder)

        nsamples = len(self.entries)
        self.total_steps = math.floor(float(nsamples) / float(self.batch_size)) if self.drop_remainder else math.ceil(
            float(nsamples) / float(self.batch_size))

        # PADDED BATCH the dataset
        if not self.even_batches:
            dataset = dataset.padded_batch(
                batch_size=self.batch_size,
                padded_shapes=(
                    # tf.TensorShape([257, None]),  # Audio
                    # tf.TensorShape([self.nfeatures, None]),  # Audio
                    tf.TensorShape([240, None]),  # Audio
                    tf.TensorShape([]),  # Input len
                    tf.TensorShape([None]),  # Label
                    tf.TensorShape([]),  # Label len
                ),
                padding_values=(
                    0.0,  # Audio
                    0.0,  # Input len
                    self.tokenizer.num_labels,  # Blank idx
                    0,  # Label len
                ),
                drop_remainder=self.drop_remainder
            )

            if self.service_ip:
                dataset = dataset.apply(tf.data.experimental.service.distribute(
                    processing_mode="distributed_epoch", service="grpc://" + self.service_ip + ":31000",
                    max_outstanding_requests=8, max_request_pipelining_per_worker=8
                ))

            if self.repeat_single_batch:

                dataset = dataset.take(1).cache().repeat(200)
                # batch = next(iter(dataset))
                # logger.info(f"BATCH SHAPE: {batch[0].shape}, {batch[1].shape}, NUM BATCHES: {num_batches}")
            # PREFETCH to improve speed of input length
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset
