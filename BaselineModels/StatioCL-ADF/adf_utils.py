import os
import time

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def update(self, amount=1):
            pass

def _require_adfuller():
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:
        raise ImportError(
            "StatioCL-ADF requires statsmodels. Install statsmodels in the "
            "training environment before running with --ssl True."
        ) from exc
    return adfuller


def _segment_raw(data, segment_len):
    sample_size, seq_len, channels = data.shape
    seg_num = seq_len // segment_len
    if seg_num == 0:
        raise ValueError(
            f"segment_len={segment_len} is longer than sequence length={seq_len}"
        )
    valid_len = seg_num * segment_len
    data = data[:, :valid_len, :]
    return np.reshape(data, (sample_size, seg_num, segment_len, channels))


def _adf_state(series, adfuller, alpha, maxlag=None, autolag="AIC", regression="c"):
    series = np.asarray(series, dtype=np.float64)
    series = series[np.isfinite(series)]
    if series.shape[0] < 3 or np.allclose(series, series[0]):
        return 0
    try:
        p_value = adfuller(
            series,
            maxlag=maxlag,
            autolag=autolag,
            regression=regression,
        )[1]
    except Exception:
        return 0
    return int(p_value > alpha)


def _majority_vote(states):
    if len(states) == 0:
        return 0
    return int(np.mean(states) >= 0.5)


def _prior_progress(total_samples):
    total_steps = 2 * int(total_samples)
    return tqdm(
        total=total_steps,
        desc="Computing ADF prior",
        unit="sample-pass",
        mininterval=5.0,
        miniters=max(total_steps // 100, 1),
        dynamic_ncols=True,
        leave=True,
    )


def _format_elapsed(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes):02d}m {seconds:05.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:05.2f}s"
    return f"{seconds:.2f}s"


def _cache_path(
    save_dir,
    dataset_name,
    fold,
    seed,
    segment_len,
    alpha,
    sample_autolag="AIC",
    sample_maxlag=None,
):
    alpha_tag = str(alpha).replace(".", "p")
    filename_stem = (
        f"adf_labels_{dataset_name}_fold{fold}_seed{seed}_"
        f"seg{segment_len}_alpha{alpha_tag}"
    )
    # Preserve compatibility with existing caches created by the original
    # default configuration. Non-default lag settings get distinct caches.
    if sample_autolag == "AIC" and sample_maxlag is None:
        filename = f"{filename_stem}.npz"
    else:
        autolag_tag = "none" if sample_autolag is None else str(sample_autolag).lower()
        autolag_tag = autolag_tag.replace("-", "")
        maxlag_tag = "auto" if sample_maxlag is None else str(sample_maxlag)
        filename = (
            f"{filename_stem}_sampleautolag{autolag_tag}_"
            f"samplemaxlag{maxlag_tag}.npz"
        )
    return os.path.join(save_dir, filename)


def _compute_sample_labels(
    data,
    adfuller,
    alpha,
    sample_autolag,
    sample_maxlag=None,
    progress=None,
):
    labels = np.zeros(data.shape[0], dtype=np.int64)
    for sample_idx in range(data.shape[0]):
        channel_states = [
            _adf_state(
                data[sample_idx, channel_idx],
                adfuller,
                alpha,
                maxlag=sample_maxlag,
                autolag=sample_autolag,
                regression="c",
            )
            for channel_idx in range(data.shape[1])
        ]
        labels[sample_idx] = _majority_vote(channel_states)
        if progress is not None:
            progress.update(1)
    return labels


def _compute_segment_labels(data, adfuller, alpha, segment_len, progress=None):
    # Match AAAI2027.models.statistic_prior._segment_modality:
    # raw samples are converted from (N, C, T) to (N, T, C), truncated to
    # full segments, then reshaped to (N, segment_num, segment_len, C).
    segments = _segment_raw(np.transpose(data, (0, 2, 1)), segment_len)
    labels = np.zeros((segments.shape[0], segments.shape[1]), dtype=np.int64)
    for sample_idx in range(segments.shape[0]):
        for segment_idx in range(segments.shape[1]):
            channel_states = [
                _adf_state(
                    segments[sample_idx, segment_idx, :, channel_idx],
                    adfuller,
                    alpha,
                    maxlag=0,
                    autolag=None,
                    regression="n",
                )
                for channel_idx in range(segments.shape[-1])
            ]
            labels[sample_idx, segment_idx] = _majority_vote(channel_states)
        if progress is not None:
            progress.update(1)
    return labels


def _print_distribution(name, labels):
    values, counts = np.unique(labels.reshape(-1), return_counts=True)
    total = max(int(np.sum(counts)), 1)
    parts = [
        f"{int(value)}={int(count)} ({float(count) / total:.4f})"
        for value, count in zip(values, counts)
    ]
    print(f"{name} ADF distribution: " + ", ".join(parts))


def _load_cached_labels(cache_path, expected_count=None):
    if not os.path.exists(cache_path):
        return None
    cached = np.load(cache_path)
    sample_adf = cached["sample_adf"].astype(np.int64)
    segment_adf = cached["segment_adf"].astype(np.int64)
    if expected_count is not None and (
        sample_adf.shape[0] != expected_count or segment_adf.shape[0] != expected_count
    ):
        print(f"Ignoring stale ADF cache with mismatched sample count: {cache_path}")
        return None
    print(f"Loaded ADF labels from {cache_path}")
    _print_distribution("Sample", sample_adf)
    _print_distribution("Segment", segment_adf)
    return sample_adf, segment_adf


def _save_labels(
    cache_path,
    sample_adf,
    segment_adf,
    segment_len,
    alpha,
    sample_autolag,
    sample_maxlag,
):
    np.savez_compressed(
        cache_path,
        sample_adf=sample_adf,
        segment_adf=segment_adf,
        segment_len=np.asarray(segment_len, dtype=np.int64),
        alpha=np.asarray(alpha, dtype=np.float32),
        sample_autolag=np.asarray(
            "none" if sample_autolag is None else sample_autolag
        ),
        sample_maxlag=np.asarray(
            -1 if sample_maxlag is None else sample_maxlag,
            dtype=np.int64,
        ),
    )
    print(f"Saved ADF labels to {cache_path}")
    _print_distribution("Sample", sample_adf)
    _print_distribution("Segment", segment_adf)


def load_or_compute_adf_labels(
    data,
    dataset_name,
    fold,
    seed,
    save_dir,
    segment_len=4,
    alpha=0.01,
    sample_autolag="AIC",
    sample_maxlag=None,
):
    os.makedirs(save_dir, exist_ok=True)
    cache_path = _cache_path(
        save_dir,
        dataset_name,
        fold,
        seed,
        segment_len,
        alpha,
        sample_autolag,
        sample_maxlag,
    )
    cached = _load_cached_labels(cache_path, expected_count=data.shape[0])
    if cached is not None:
        return cached

    adfuller = _require_adfuller()
    print(
        "Computing ADF labels "
        f"(alpha={alpha}, segment_len={segment_len}, "
        f"sample_autolag={sample_autolag}, sample_maxlag={sample_maxlag})"
    )
    start_time = time.perf_counter()
    with _prior_progress(data.shape[0]) as progress:
        sample_adf = _compute_sample_labels(
            data,
            adfuller,
            alpha,
            sample_autolag,
            sample_maxlag,
            progress=progress,
        )
        segment_adf = _compute_segment_labels(
            data,
            adfuller,
            alpha,
            segment_len,
            progress=progress,
        )
    elapsed = time.perf_counter() - start_time
    print(
        f"Total ADF prior computation time: {_format_elapsed(elapsed)} "
        f"({elapsed:.2f} seconds)"
    )
    _save_labels(
        cache_path,
        sample_adf,
        segment_adf,
        segment_len,
        alpha,
        sample_autolag,
        sample_maxlag,
    )
    return sample_adf, segment_adf


def load_or_compute_adf_labels_streaming(
    path_name,
    train_idx,
    dataset_name,
    fold,
    seed,
    save_dir,
    segment_len=4,
    alpha=0.01,
    sample_autolag="AIC",
    sample_maxlag=None,
):
    os.makedirs(save_dir, exist_ok=True)
    cache_path = _cache_path(
        save_dir,
        dataset_name,
        fold,
        seed,
        segment_len,
        alpha,
        sample_autolag,
        sample_maxlag,
    )

    expected_count = 0
    for subject_id in train_idx:
        for data_path, _ in path_name[subject_id]:
            expected_count += int(np.load(data_path, mmap_mode="r").shape[0])

    cached = _load_cached_labels(cache_path, expected_count=expected_count)
    if cached is not None:
        return cached

    adfuller = _require_adfuller()
    print(
        "Computing streamed ADF labels "
        f"(alpha={alpha}, segment_len={segment_len}, "
        f"sample_autolag={sample_autolag}, sample_maxlag={sample_maxlag})"
    )

    sample_chunks = []
    segment_chunks = []
    file_items = [
        (subject_id, data_path)
        for subject_id in train_idx
        for data_path, _ in path_name[subject_id]
    ]
    start_time = time.perf_counter()
    with _prior_progress(expected_count) as progress:
        for _, data_path in file_items:
            data = np.load(data_path, mmap_mode="r")
            sample_chunks.append(
                _compute_sample_labels(
                    data,
                    adfuller,
                    alpha,
                    sample_autolag,
                    sample_maxlag,
                    progress=progress,
                )
            )
            segment_chunks.append(
                _compute_segment_labels(
                    data,
                    adfuller,
                    alpha,
                    segment_len,
                    progress=progress,
                )
            )
            del data
    elapsed = time.perf_counter() - start_time
    print(
        f"Total ADF prior computation time: {_format_elapsed(elapsed)} "
        f"({elapsed:.2f} seconds)"
    )

    if not sample_chunks:
        raise ValueError("No pretraining files were found for streamed ADF cache computation")

    sample_adf = np.concatenate(sample_chunks, axis=0)
    segment_adf = np.concatenate(segment_chunks, axis=0)
    _save_labels(
        cache_path,
        sample_adf,
        segment_adf,
        segment_len,
        alpha,
        sample_autolag,
        sample_maxlag,
    )
    return sample_adf, segment_adf
