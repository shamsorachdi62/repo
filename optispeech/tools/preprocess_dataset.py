import argparse
import csv
import json
import os
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import hydra
import numpy as np
import rootutils
from hydra import compose, initialize
from tqdm import tqdm
from omegaconf import OmegaConf

from optispeech.dataset import TextWavDataset, do_preprocess_utterance
from optispeech.utils import get_script_logger


log = get_script_logger(__name__)


_worker_feature_extractor = None
_worker_text_processor = None
_worker_wav_path = None
_worker_data_dir = None
_worker_sids = None
_worker_lids = None
_worker_threadpool_limits = None


def init_worker(text_processor_cfg, feature_extractor_cfg, wav_path, data_dir, sids, lids):
    global _worker_feature_extractor
    global _worker_text_processor
    global _worker_wav_path
    global _worker_data_dir
    global _worker_sids
    global _worker_lids
    global _worker_threadpool_limits

    # Each worker is already a process. Keep per-library CPU thread pools from
    # multiplying into n_workers * n_threads oversubscription.
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        from threadpoolctl import threadpool_limits

        _worker_threadpool_limits = threadpool_limits(limits=1)
    except Exception:
        pass

    _worker_text_processor = hydra.utils.instantiate(text_processor_cfg)
    _worker_feature_extractor = hydra.utils.instantiate(feature_extractor_cfg)
    _worker_feature_extractor.initialize_components()
    _worker_wav_path = Path(wav_path)
    _worker_data_dir = Path(data_dir)
    _worker_sids = sids
    _worker_lids = lids


def process_row(row):
    if len(row) == 2:
        filestem, text = row
        speaker = lang = None
    elif len(row) == 3:
        filestem, speaker, text = row
        lang = None
    elif len(row) == 4:
        filestem, speaker, lang, text = row
    else:
        log.error(f"Invalid number of data items in dataset row: {len(row)}")
        exit(1)
    audio_path = _worker_wav_path.joinpath(filestem + ".wav")
    audio_path = audio_path.resolve()
    sid = _worker_sids.index(speaker.strip().lower()) if speaker else None
    lid = _worker_lids.index(lang.strip().lower()) if lang else None
    out_arrays = _worker_data_dir.joinpath(audio_path.stem + ".npz")
    out_json = _worker_data_dir.joinpath(audio_path.stem + ".json")
    if out_arrays.is_file() and out_json.is_file() and out_arrays.stat().st_size > 500:
        return audio_path.stem, True
    try:
        data = do_preprocess_utterance(
            feature_extractor=_worker_feature_extractor,
            text_processor=_worker_text_processor,
            audio_filepath=audio_path,
            text=text,
            lang=lang,
        )
    except Exception as e:
        formatted_exception = traceback.format_exception(e)
        return filestem, Exception(f"Failed to process file: {audio_path.name}", formatted_exception)
    else:
        write_data(_worker_data_dir, audio_path.stem, data, sid, lid)
        return audio_path.stem, True


def collect_results(iterator, total, data_dir):
    out_filelist = []
    for (filestem, retval) in tqdm(iterator, total=total, desc="processing", unit="utterance"):
        if isinstance(retval, Exception):
            log.error(
                f"Failed to process item {filestem}. Error: {retval.args[0]}.\n"
                f"Caused by: {''.join(retval.args[1])}"
            )
        else:
            out_filelist.append(data_dir.joinpath(filestem))
    return out_filelist


def write_data(data_dir, file_stem, data, sid, lid):
    output_file = data_dir.joinpath(file_stem)
    out_arrays = output_file.with_suffix(".npz")
    out_json = output_file.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as file:
        ph_text_data = {
            "phoneme_ids": data["phoneme_ids"],
            "text": data["text"],
        }
        if sid is not None:
            ph_text_data["sid"] = sid
        if lid is not None:
            ph_text_data["lid"] = lid
        json.dump(ph_text_data, file, ensure_ascii=False)
    np.savez(
        out_arrays,
        allow_pickle=False,
        wav=data["wav"],
        mel=data["mel"],
        energy=data["energy"],
        pitch=data["pitch"],
    )


def get_sids_and_lids(dataset, all_utterances):
    assert dataset.num_speakers >= 1, "Illogical number of speakers in the dataset"
    sids = lids = None
    if dataset.num_speakers > 1:
        row_len = len(all_utterances[0])
        assert row_len > 2, f"Speaker ID column not included. Invalid number of data items in dataset rows: {row_len}"
        sids = [sid.strip().lower() for sid in [ut[1] for ut in all_utterances]]
        assert all(sids), "Invalid input. Some utterances lack speaker identifier."
        sids = sort_by_most_common(sids)
    if dataset.text_processor.is_multi_language:
        row_len = len(all_utterances[0])
        assert row_len > 3, f"Language column not included. Invalid number of data items in dataset rows: {row_len}"
        lids = [lid.strip().lower() for lid in [ut[2] for ut in all_utterances]]
        assert all(lids), "Invalid input. Some utterances lack language identifier."
        lids = sort_by_most_common(lids)
    return sids, lids


def sort_by_most_common(iterable):
    counter = Counter(iterable)
    return [j for j, k in counter.most_common()]


def main():
    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="dataset config relative to `configs/data/` (without the suffix)",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="original data directory",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to write datafiles + train.txt and val.txt",
    )
    parser.add_argument(
        "--format",
        choices=["ljspeech"],
        default="ljspeech",
        help="Dataset format.",
    )
    parser.add_argument(
        "-w",
        "--n-workers",
        type=int,
        default=cpu_count() // 2,
        help="Number of worker processes to use",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../../configs/data"):
        cfg = compose(config_name=args.dataset)
        cfg["seed"] = 1234
    text_processor = hydra.utils.instantiate(cfg.text_processor)
    feature_extractor = hydra.utils.instantiate(cfg.feature_extractor)
    dataset = TextWavDataset(
        num_speakers=cfg.num_speakers,
        filelist_path=os.devnull,
        text_processor=text_processor,
        feature_extractor=feature_extractor,
    )

    if args.format != "ljspeech":
        log.error(f"Unsupported dataset format `{args.format}`")
        exit(1)

    data_root = Path(args.input_dir)
    # get all utterances to calculate number of speakers/languages
    all_utterances = []
    with open(data_root.joinpath("train.csv"), encoding="utf-8") as cfile:
        lines = cfile.read().splitlines()
        splitted_uts = [
            line.strip().split("|", 1)
            for line in lines
            if line.strip()
        ]
        all_utterances.extend(splitted_uts)
    with open(data_root.joinpath("val.csv"), encoding="utf-8") as cfile:
        lines = cfile.read().splitlines()
        splitted_uts = [
            line.strip().split("|", 1)
            for line in lines
            if line.strip()
        ]
        all_utterances.extend(splitted_uts)
    sids, lids = get_sids_and_lids(dataset, all_utterances)
    # Start
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir.joinpath("data")
    data_dir.mkdir(exist_ok=True)
    # eSpeak uses global state for language.
    # Comment this line if you're not using eSpeak for phonemization
    n_workers = args.n_workers if not text_processor.is_multi_language else 1
    chunk_size = max(1, args.batch_size)
    wav_path = data_root.joinpath("wav")
    text_processor_cfg = OmegaConf.to_container(cfg.text_processor, resolve=True)
    feature_extractor_cfg = OmegaConf.to_container(cfg.feature_extractor, resolve=True)
    for out_filename in ["train.txt", "val.txt"]:
        data_split = out_filename.split('.')[0]
        log.info(f"Extracting datasplit `{data_split}`")
        with open(data_root.joinpath(f"{data_split}.csv"), encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="|")
            inrows = list(reader)
        log.info(f"Found {len(inrows)} utterances in file.")
        worker_args = (text_processor_cfg, feature_extractor_cfg, wav_path, data_dir, sids, lids)
        if n_workers == 1:
            init_worker(*worker_args)
            iterator = map(process_row, inrows)
            out_filelist = collect_results(iterator, len(inrows), data_dir)
        else:
            log.info(f"Using {n_workers} worker processes with chunksize {chunk_size}")
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=init_worker,
                initargs=worker_args,
            ) as executor:
                iterator = executor.map(process_row, inrows, chunksize=chunk_size)
                out_filelist = collect_results(iterator, len(inrows), data_dir)
        out_txt = output_dir.joinpath(out_filename)
        with open(out_txt, "w", encoding="utf-8", newline="\n") as file:
            filelist = [os.fspath(fn.resolve()) for fn in out_filelist]
            file.write("\n".join(filelist))
        log.info(f"Wrote file: {out_txt}")

    # write speaker-ids and language-ids
    if sids is not None:
        sids_json = output_dir.joinpath("speaker_ids.json")
        with open(sids_json, "w", encoding="utf-8") as jfile:
            json.dump(sids, jfile, ensure_ascii=False, indent=2)
        log.info(f"Wrote speaker IDs to file: {sids_json}")
    if lids is not None:
        lids_json = output_dir.joinpath("language_ids.json")
        with open(lids_json, "w", encoding="utf-8") as jfile:
            json.dump(lids, jfile, ensure_ascii=False, indent=2)
        log.info(f"Wrote language IDs to file: {lids_json}")
    log.info("Process done!")


if __name__ == "__main__":
    main()
