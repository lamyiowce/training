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
import tensorflow_io as tfio

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


class Vectorizer:
    def __init__(self, max_len, labels):
        self.vocab = labels
        self.max_len = max_len
        self.char_to_idx = {}
        self.reported = set()
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i
        self.pad_value = 0

    def parse_transcript(self, text):
        text = text.upper()
        # text = tf.strings.lower(text)
        text = text[: self.max_len - 2]
        # text = tf.strings.join(["<", text, ">"])
        text = "<" + text + ">"
        # pad_len = self.max_len - len(text)
        for ch in text:
            if ch not in self.char_to_idx and ch not in self.reported:
                print(f"CHAR: '{ch}' (ascii: {ord(ch)}) not in dictionary.")
                self.reported.add(ch)

        return [self.char_to_idx.get(ch, 1) for ch in text]  # + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


class SnapshotDataset:
    def __init__(self,
                 pipeline,
                 caching_period,
                 vectorizer,
                 max_audio_len,
                 data_paths,
                 snapshot_path=None,
                 num_elems_to_load=None,
                 service_ip=None,
                 wav=False,
                 repeat_single_batch=False,
                 even_batches=False,
                 drop_remainder=False,
                 ):
        self.data_paths = data_paths
        self.drop_remainder = drop_remainder
        self.max_audio_len = max_audio_len
        if 'snapshot' in pipeline:
            assert snapshot_path
        self.pipeline = pipeline
        self.caching_period = caching_period
        self.vectorizer = vectorizer
        self.snapshot_path = snapshot_path
        self.repeat_single_batch = repeat_single_batch
        self.num_elems_to_load = num_elems_to_load  # if not repeat_single_batch else 64 # Assuming this will be > than the batch size
        self.service_ip = service_ip
        self.wav = wav
        self.even_batches = even_batches
        self.mel_bins = 161

    def read_entries(self):
        self.entries = []
        logger = tf.get_logger()
        for file_path in self.data_paths:
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()[1:]
                limit = self.num_elems_to_load or len(temp_lines)
                # Skip the header of tsv file
                self.entries += temp_lines[:limit]
        # The files is "\t" seperated
        self.entries = [line.split("\t", 2) for line in self.entries]
        for i, line in enumerate(self.entries):
            self.entries[i][-1] = " ".join([str(x) for x in self.vectorizer.parse_transcript(line[-1])])
        self.entries = np.array(self.entries)
        self.total_steps = len(self.entries)

    @staticmethod
    def load(record: tf.Tensor):
        path = record[0]
        audio = tfio.audio.AudioIOTensor(path, dtype=tf.int16)
        audio = tf.cast(audio.to_tensor(), tf.float32) / 32768.0
        audio = tf.squeeze(audio, axis=-1)  # Make shape (LEN,)
        return audio, record[2]

    @staticmethod
    def load_wav(record: tf.Tensor):
        path = record[0]
        audio, _rate = tf.audio.decode_wav(tf.io.read_file(path))
        audio = tf.squeeze(audio, axis=-1)  # Make shape (LEN,)
        return audio, record[2]

    def _preprocess_audio(self, audio: tf.Tensor):
        # Preprocessing hyperparameters from Lingvo:
        # https://github.com/tensorflow/lingvo/blob/master/lingvo/tools/audio_lib.py
        FRAME_STEP_SECONDS = 10. / 1000
        FRAME_LENGTH_SECONDS = 25. / 1000
        MEL_BINS = self.mel_bins
        MEL_LOWER_BOUND_HERTZ = 125.
        MEL_UPPER_BOUND_HERTZ = 7600.
        RATE = 16000
        # TODO: Maybe apply preemphasis.

        # spectrogram using stft
        frame_len = tf.cast(tf.math.round(FRAME_LENGTH_SECONDS * RATE), tf.int32)
        frame_step = tf.cast(tf.math.round(FRAME_STEP_SECONDS * RATE), tf.int32)

        x = tf.abs(tf.signal.stft(audio, frame_length=frame_len, frame_step=frame_step, fft_length=512))
        x = tf.math.pow(x, 0.5)
        x = tfio.audio.melscale(
            x, rate=RATE, mels=MEL_BINS, fmin=MEL_LOWER_BOUND_HERTZ, fmax=MEL_UPPER_BOUND_HERTZ)
        x = tfio.audio.dbscale(x, top_db=80)

        # Standardization. TODO: mean and std should be calculated across samples.
        means = tf.math.reduce_mean(x, 1, keepdims=True)
        stddevs = tf.math.reduce_std(x, 1, keepdims=True)
        x = (x - means) / stddevs
        return x

    def create_inputs(self, audio, indices):
        # Audio shape (None, 80)
        audio = tf.transpose(audio)
        # Audio shape (None, 80)
        audio = tf.reshape(audio, shape=[1, audio.shape[0], -1])
        # Audio shape (1, 80, None)

        input_length = tf.cast(tf.shape(audio)[0], tf.int32)
        print(audio.shape)
        label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32)
        label_length = tf.cast(tf.shape(label)[0], tf.int32)
        # Inputs, Targets, inputs percentage of max len, target len
        return audio, label, input_length, label_length

    def pad_audio(self, x, indices):
        length = self.max_audio_len
        paddings = tf.constant([[0, length], [0, 0], [0, 0]])
        x = tf.pad(x, paddings, "CONSTANT")[:length, :, :]
        x = tf.ensure_shape(x, [length, 80])

        indices_length = self.vectorizer.max_length
        indices_paddings = tf.constant([[0, indices_length]])
        indices = tf.pad(tf.strings.split(indices), indices_paddings, "CONSTANT", constant_values="0")[
                  :indices_length]
        indices = tf.ensure_shape(indices, [indices_length])
        return x, indices

    def create_tf_dataset(self, batch_size: int, specaugment_config=None):
        self.read_entries()

        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        if self.wav:
            dataset = dataset.map(self.load_wav, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(self.load, num_parallel_calls=tf.data.AUTOTUNE)
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
                    lambda d: {"source": spec_augment(d["source"], time_warping_para=80, frequency_masking_para=27,
                                                      time_masking_para=100),
                               "target": d["target"]},
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
            dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=self.drop_remainder)

        nsamples = len(self.entries)
        self.total_steps = math.floor(float(nsamples) / float(batch_size)) if self.drop_remainder else math.ceil(
            float(nsamples) / float(batch_size))

        # PADDED BATCH the dataset
        if not self.even_batches:
            dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=(
                    tf.TensorShape([1, self.mel_bins, None]),  # Audio
                    tf.TensorShape([None]),  # Label
                    tf.TensorShape([]),  # Input len
                    tf.TensorShape([]),  # Label len
                ),
                padding_values=(
                    0.0,  # Audio
                    self.vectorizer.pad_value,  # Label
                    0,  # Input len
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
                num_batches = len(self.entries) // batch_size
                dataset = dataset.take(1).cache().repeat(num_batches)
                # batch = next(iter(dataset))
                # logger.info(f"BATCH SHAPE: {batch[0].shape}, {batch[1].shape}, NUM BATCHES: {num_batches}")
            # PREFETCH to improve speed of input length
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset
