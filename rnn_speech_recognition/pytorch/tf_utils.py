import tensorflow as tf


@tf.function
def melscale(input, rate, mels, fmin=0.0, fmax=None, name=None):
    """
    Turn spectrogram into mel scale spectrogram

    Args:
      input: A spectrogram Tensor with shape [frames, nfft+1].
      rate: Sample rate of the audio.
      mels: Number of mel filterbanks.
      fmin: Minimum frequency.
      fmax: Maximum frequency.
      name: A name for the operation (optional).

    Returns:
      A tensor of mel spectrogram with shape [frames, mels].
    """

    if fmax is None:
        fmax = rate / 2.
    nbins = tf.shape(input)[-1]
    matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mels,
        num_spectrogram_bins=nbins,
        sample_rate=rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )

    return tf.tensordot(input, matrix, 1)


@tf.function
def dbscale(input, top_db, name=None):
    """
    Turn spectrogram into db scale

    Args:
      input: A spectrogram Tensor.
      top_db: Minimum negative cut-off `max(10 * log10(S)) - top_db`
      name: A name for the operation (optional).

    Returns:
      A tensor of mel spectrogram with shape [frames, mels].
    """

    power = tf.math.square(input)
    log_spec = 10.0 * (tf.math.log(power) / tf.math.log(10.0))
    log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)
    return log_spec


def stack_subsample_frames(x, stacking=1, subsampling=1):
    """ Stacks frames together across feature dim, and then subsamples

     x.shape: FEAT, TIME
    output FEAT * stacking, TIME / subsampling

    """
    # x.shape: FEAT, TIME
    seq = []
    x_len = tf.shape(x)[1]
    for n in range(0, stacking):
        tmp = x[:, n:x_len - stacking + 1 + n:subsampling]
        seq.append(tmp)
    print(seq)
    x = tf.concat(seq, axis=0)
    return x
