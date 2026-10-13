#!/usr/bin/env python3
"""Download pinned experiment assets; never select/transform experimental samples.

Deployment helper only. Uses mirror URLs for repository assets and the original
author release hosts for ProofWriter/SAMSum. Existing experimental code is untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import http.client
import importlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

# Replaced at release time with the independently checked source manifests.
MANIFEST = json.loads('{"schema_version": 1, "generated_utc": "2026-10-11", "download_policy": "Pinned official repository files through hf-mirror.com; original author archive host for ProofWriter and SAMSum. No Hugging Face token or repository API call is required to download listed URLs.", "datasets": [{"key": "gsm8k", "repo_id": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866", "main_source_revision": "740312add88f781978c0658806c59bc2815b9866", "config": "main", "files": [{"path": "main/test-00000-of-00001.parquet", "size": 419088, "lfs": {"sha256": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59", "size": 419088, "pointerSize": 131}, "blobId": "861005fb045993f46d198aa797440f4096f9aa17", "split": "test", "expected_rows": 1319, "url": "https://hf-mirror.com/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet", "canonical_url": "https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet"}, {"path": "main/train-00000-of-00001.parquet", "size": 2306545, "lfs": {"sha256": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d", "size": 2306545, "pointerSize": 132}, "blobId": "61b47c916c27533feaa3763484ae0ea58f92b056", "split": "train", "expected_rows": 7473, "url": "https://hf-mirror.com/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet", "canonical_url": "https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"}], "format": "parquet", "expected_rows": {"train": 7473, "test": 1319}, "metadata_evidence_url": "https://huggingface.co/api/datasets/openai/gsm8k/revision/740312add88f781978c0658806c59bc2815b9866?blobs=true", "dataset_card_url": "https://huggingface.co/datasets/openai/gsm8k"}, {"key": "qqp", "repo_id": "nyu-mll/glue", "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "main_source_revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "config": "qqp", "files": [{"path": "qqp/test-00000-of-00001.parquet", "size": 36694152, "lfs": {"sha256": "95d5d1efcfa3ff7e090565e98085770b3497aad8dbcf12996412b23d2fb669e8", "size": 36694152, "pointerSize": 133}, "blobId": "15f49242dcc3aec29e9f44ef8976ef15e5957f4b", "split": "test", "expected_rows": 390965, "url": "https://hf-mirror.com/datasets/nyu-mll/glue/resolve/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/qqp/test-00000-of-00001.parquet", "canonical_url": "https://huggingface.co/datasets/nyu-mll/glue/resolve/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/qqp/test-00000-of-00001.parquet"}, {"path": "qqp/train-00000-of-00001.parquet", "size": 33558839, "lfs": {"sha256": "4d6f02e643f7c36e9a4f7d4971a5ee9bd74063a319452fe6c87850c739774cd7", "size": 33558839, "pointerSize": 133}, "blobId": "d8c3b92d9dad49c730e27e8ca80ad7781cee338d", "split": "train", "expected_rows": 363846, "url": "https://hf-mirror.com/datasets/nyu-mll/glue/resolve/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/qqp/train-00000-of-00001.parquet", "canonical_url": "https://huggingface.co/datasets/nyu-mll/glue/resolve/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/qqp/train-00000-of-00001.parquet"}, {"path": "qqp/validation-00000-of-00001.parquet", "size": 3729274, "lfs": {"sha256": "efd86a539c412d74874ee451573d7bd142f56c47fe36de033b9f367d8bb0fa71", "size": 3729274, "pointerSize": 132}, "blobId": "d5fe977f2e937bf4895b5ae533c18a4e5f292771", "split": "validation", "expected_rows": 40430, "url": "https://hf-mirror.com/datasets/nyu-mll/glue/resolve/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/qqp/validation-00000-of-00001.parquet", "canonical_url": "https://huggingface.co/datasets/nyu-mll/glue/resolve/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/qqp/validation-00000-of-00001.parquet"}], "format": "parquet", "expected_rows": {"train": 363846, "validation": 40430, "test": 390965}, "metadata_evidence_url": "https://huggingface.co/api/datasets/nyu-mll/glue/revision/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c?blobs=true", "dataset_card_url": "https://huggingface.co/datasets/nyu-mll/glue", "evaluation_note": "test labels are hidden (-1). Use validation as labeled evaluation; retain test as official unlabeled split and never invent labels."}, {"key": "mbpp", "repo_id": "google-research-datasets/mbpp", "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "main_source_revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "config": "full", "files": [{"path": "full/prompt-00000-of-00001.parquet", "size": 7878, "lfs": {"sha256": "a053e4bb85ceb77430ae80592addb4ca4dc6ba087592f9e04537800ee88b7431", "size": 7878, "pointerSize": 129}, "blobId": "ea0cd63b33b4e2e09cd3c3098971455a07e8ffe2", "split": "prompt", "expected_rows": 10, "url": "https://hf-mirror.com/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/prompt-00000-of-00001.parquet", "canonical_url": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/prompt-00000-of-00001.parquet"}, {"path": "full/test-00000-of-00001.parquet", "size": 115824, "lfs": {"sha256": "566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77", "size": 115824, "pointerSize": 131}, "blobId": "f9bdcec1eeec1f401ce091800fd69fc517d5920f", "split": "test", "expected_rows": 500, "url": "https://hf-mirror.com/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/test-00000-of-00001.parquet", "canonical_url": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/test-00000-of-00001.parquet"}, {"path": "full/train-00000-of-00001.parquet", "size": 87223, "lfs": {"sha256": "09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae", "size": 87223, "pointerSize": 130}, "blobId": "f2b0d23528ec7e7b46092b8dc91cbb3e86fe598f", "split": "train", "expected_rows": 374, "url": "https://hf-mirror.com/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/train-00000-of-00001.parquet", "canonical_url": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/train-00000-of-00001.parquet"}, {"path": "full/validation-00000-of-00001.parquet", "size": 25144, "lfs": {"sha256": "3f0ec060987432d99fe8fb409d31e6c67445b208a01741c5583517c80a10fe80", "size": 25144, "pointerSize": 130}, "blobId": "e14b00046fdd492f40859693aa800cf7ca666a61", "split": "validation", "expected_rows": 90, "url": "https://hf-mirror.com/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/validation-00000-of-00001.parquet", "canonical_url": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/validation-00000-of-00001.parquet"}], "format": "parquet", "expected_rows": {"train": 374, "validation": 90, "test": 500, "prompt": 10}, "metadata_evidence_url": "https://huggingface.co/api/datasets/google-research-datasets/mbpp/revision/4bb6404fdc6cacfda99d4ac4205087b89d32030c?blobs=true", "dataset_card_url": "https://huggingface.co/datasets/google-research-datasets/mbpp", "task_id_ranges": {"train": [601, 974], "validation": [511, 600], "test": [11, 510], "prompt": [1, 10]}}, {"key": "iwslt2017", "repo_id": "IWSLT/iwslt2017", "revision": "9ca48082a7aae6d0a9d18efb292c50eacd10d2e8", "main_source_revision": "c18a4f81a47ae6fa079fe9d32db288ddde38451d", "config": "iwslt2017-en-de", "files": [{"path": "iwslt2017-en-de/test/0000.parquet", "size": 1002104, "lfs": {"sha256": "3b06faa5a242e8588f82244cf06753c3788520b2fd109f8cb513b70ee7234b3d", "size": 1002104, "pointerSize": 132}, "blobId": "22a4f295e5b25526d6b5608e3e3b7090d466a18e", "split": "test", "expected_rows": 8079, "url": "https://hf-mirror.com/datasets/IWSLT/iwslt2017/resolve/9ca48082a7aae6d0a9d18efb292c50eacd10d2e8/iwslt2017-en-de/test/0000.parquet", "canonical_url": "https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/9ca48082a7aae6d0a9d18efb292c50eacd10d2e8/iwslt2017-en-de/test/0000.parquet"}, {"path": "iwslt2017-en-de/train/0000.parquet", "size": 26415796, "lfs": {"sha256": "39f518537ee06492a56f01366044454f14fd8dc50434f74181905dead0b4bd2a", "size": 26415796, "pointerSize": 133}, "blobId": "20daccdee0e8484af71c0f524dce54b4d992bf05", "split": "train", "expected_rows": 206112, "url": "https://hf-mirror.com/datasets/IWSLT/iwslt2017/resolve/9ca48082a7aae6d0a9d18efb292c50eacd10d2e8/iwslt2017-en-de/train/0000.parquet", "canonical_url": "https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/9ca48082a7aae6d0a9d18efb292c50eacd10d2e8/iwslt2017-en-de/train/0000.parquet"}, {"path": "iwslt2017-en-de/validation/0000.parquet", "size": 132162, "lfs": {"sha256": "fdaa92327a2780b7c2bd44768870aafbd1fa4e3eda4312292319bf06088f059a", "size": 132162, "pointerSize": 131}, "blobId": "7d1cd2fbd65283d82429cbecada787d552bf44ee", "split": "validation", "expected_rows": 888, "url": "https://hf-mirror.com/datasets/IWSLT/iwslt2017/resolve/9ca48082a7aae6d0a9d18efb292c50eacd10d2e8/iwslt2017-en-de/validation/0000.parquet", "canonical_url": "https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/9ca48082a7aae6d0a9d18efb292c50eacd10d2e8/iwslt2017-en-de/validation/0000.parquet"}], "format": "parquet", "expected_rows": {"train": 206112, "validation": 888, "test": 8079}, "metadata_evidence_url": "https://huggingface.co/api/datasets/IWSLT/iwslt2017/revision/9ca48082a7aae6d0a9d18efb292c50eacd10d2e8?blobs=true", "dataset_card_url": "https://huggingface.co/datasets/IWSLT/iwslt2017", "revision_ref": "refs/convert/parquet", "source_revision_note": "main_source_revision is the source repository tree observed on 2026-10-11, not an assertion of the historical input commit of the 2024 conversion. Files are the complete non-partial conversion under the original official repository; validate expected rows and schema before use.", "raw_source_archive": {"path": "data/2017-01-trnted/texts/en/de/en-de.zip", "size": 16758334, "sha256": "d64c16ea33b6d49c44ff5f11768f44a382ed1c0cba351a00c776f227fad80e39", "url": "https://hf-mirror.com/datasets/IWSLT/iwslt2017/resolve/c18a4f81a47ae6fa079fe9d32db288ddde38451d/data/2017-01-trnted/texts/en/de/en-de.zip"}}], "archives": [{"key": "proofwriter", "source_identity": "AllenAI ProofWriter V2020.12.3 original release", "homepage": "https://allenai.org/data/proofwriter", "url": "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip", "filename": "proofwriter-dataset-V2020.12.3.zip", "size": 214185889, "etag": "abe79e94eb3205ff0978c88eeffaeedc-26", "sha256": null, "accept_ranges": "bytes", "format": "zip", "members": [{"path": "proofwriter-dataset-V2020.12.3/OWA/depth-0/meta-train.jsonl", "size": 120501114, "split": "train", "depth": 0, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-0/meta-dev.jsonl", "size": 17179315, "split": "dev", "depth": 0, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-0/meta-test.jsonl", "size": 34297241, "split": "test", "depth": 0, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-1/meta-train.jsonl", "size": 66877238, "split": "train", "depth": 1, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-1/meta-dev.jsonl", "size": 9885810, "split": "dev", "depth": 1, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-1/meta-test.jsonl", "size": 19338142, "split": "test", "depth": 1, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-2/meta-train.jsonl", "size": 68187947, "split": "train", "depth": 2, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-2/meta-dev.jsonl", "size": 9692273, "split": "dev", "depth": 2, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-2/meta-test.jsonl", "size": 19488952, "split": "test", "depth": 2, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-3/meta-train.jsonl", "size": 72168302, "split": "train", "depth": 3, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-3/meta-dev.jsonl", "size": 10367461, "split": "dev", "depth": 3, "world": "OWA"}, {"path": "proofwriter-dataset-V2020.12.3/OWA/depth-3/meta-test.jsonl", "size": 20902607, "split": "test", "depth": 3, "world": "OWA"}], "selection_rule": "Extract exactly the 12 member paths listed under members. This is OWA depth directories 0,1,2,3 and exact filenames meta-train.jsonl, meta-dev.jsonl, meta-test.jsonl. Do not select meta-abduct-* or meta-stage-* and do not conflate directory depth with per-question QDep. Keep the downloaded ZIP intact.", "layout": "proofwriter/proofwriter-dataset-V2020.12.3/OWA/depth-{0,1,2,3}/meta-{train,dev,test}.jsonl", "verification_note": "HEAD returned 200 and Accept-Ranges bytes. ZIP central directory verified using a 65536-byte HTTP range request; member uncompressed sizes are from the official ZIP central directory. ETag is S3 multipart and is not a SHA256; compute and record local SHA256 after complete download. Exact theory/question counts remain to be measured after extraction."}, {"key": "samsum", "source_identity": "SAMSum original author arXiv v2 ancillary corpus.7z", "homepage": "https://arxiv.org/abs/1911.12237v2", "url": "https://arxiv.org/src/1911.12237v2/anc/corpus.7z", "filename": "corpus.7z", "size": 2944100, "etag": "sha256:57947d3dbdbba32300b85b33f0c88e59abdbeb4ba838246d26287c60b3a817ef", "sha256": "a97674c66726f66b98a08ca5e8868fb8af9d4843f2b05c4f839bc5cfe91e8899", "accept_ranges": "bytes", "format": "7z", "expected_rows": {"train": 14732, "validation": 818, "test": 819}, "selection_rule": "Extract corpus.7z safely. Retain all files including original licence/readme. Identify official train, validation/val, and test JSON files from the archive listing. Use train and test for configured training/evaluation and retain validation. Do not use third-party reformatted copies.", "layout": "samsum/ preserving archive-relative paths", "verification_note": "Complete 2944100-byte original arXiv v2 payload independently fetched and SHA256 measured twice on 2026-10-11. The response ETag sha256:57947d... is NOT the SHA256 of the 7z payload; retain it only as HTTP header evidence. Original train/val/test counts and schema verified by in-memory libarchive extraction; train retains original empty-dialogue sample ID 13828807.", "member_sha256": {"test.json": "eb5ac232d04982fdf70766b10bd07223732d8de0b817a0ab2a30a79ea99a693a", "train.json": "c0df0898ad4322462cb73959f0ca820b31c4dc27e44b1ef64fbc34f7e12f1cb5", "val.json": "f291b34ea2a851c4fd67bc9e1c05003f0dde2c2ae1a501ba733b9d97b6166349"}}], "models": [{"key": "qwen3_1_7b", "repo_id": "Qwen/Qwen3-1.7B", "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "modelscope_repo": "Qwen/Qwen3-1.7B", "modelscope_revision": "4855588ea1a12789f2e965e5f52a9e4a24c94b2a", "files": [{"path": "config.json", "size": 726, "sha256": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/config.json", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json"]}, {"path": "generation_config.json", "size": 239, "sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/generation_config.json", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/generation_config.json"]}, {"path": "merges.txt", "size": 1671853, "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/merges.txt", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/merges.txt"]}, {"path": "model-00001-of-00002.safetensors", "size": 3441185608, "sha256": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/model-00001-of-00002.safetensors", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/model-00001-of-00002.safetensors"]}, {"path": "model-00002-of-00002.safetensors", "size": 622329984, "sha256": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/model-00002-of-00002.safetensors", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/model-00002-of-00002.safetensors"]}, {"path": "model.safetensors.index.json", "size": 25605, "sha256": "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/model.safetensors.index.json", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/model.safetensors.index.json"]}, {"path": "tokenizer.json", "size": 11422654, "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/tokenizer.json", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/tokenizer.json"]}, {"path": "tokenizer_config.json", "size": 9732, "sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/tokenizer_config.json", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/tokenizer_config.json"]}, {"path": "vocab.json", "size": 2776833, "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/vocab.json", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/vocab.json"]}, {"path": "README.md", "size": 13963, "sha256": "257e52c419dac2258852643f18af6c974f21f8c6c1b6f371b6cca6201cf29091", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/README.md", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md"]}, {"path": "LICENSE", "size": 11343, "sha256": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e", "urls": ["https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/4855588ea1a12789f2e965e5f52a9e4a24c94b2a/LICENSE", "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/LICENSE"]}]}, {"key": "all_minilm_l6_v2", "repo_id": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "modelscope_repo": "AI-ModelScope/all-MiniLM-L6-v2", "modelscope_revision": "ec79a486dac00511c26bf3a04531c2c5147e723a", "files": [{"path": "1_Pooling/config.json", "size": 190, "sha256": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/1_Pooling/config.json", "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/1_Pooling/config.json"]}, {"path": "config.json", "size": 612, "sha256": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/config.json", "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/config.json"]}, {"path": "config_sentence_transformers.json", "size": 116, "sha256": "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/config_sentence_transformers.json", "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/config_sentence_transformers.json"]}, {"path": "model.safetensors", "size": 90868376, "sha256": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/model.safetensors", "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors"]}, {"path": "modules.json", "size": 349, "sha256": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/modules.json", "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/modules.json"]}, {"path": "sentence_bert_config.json", "size": 53, "sha256": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/sentence_bert_config.json", "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/sentence_bert_config.json"]}, {"path": "special_tokens_map.json", "size": 112, "sha256": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/special_tokens_map.json", "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/special_tokens_map.json"]}, {"path": "tokenizer.json", "size": 466247, "sha256": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/tokenizer.json", "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json"]}, {"path": "tokenizer_config.json", "size": 350, "sha256": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/tokenizer_config.json", "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer_config.json"]}, {"path": "vocab.txt", "size": 231508, "sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/vocab.txt", "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/vocab.txt"]}, {"path": "README.md", "size": 10659, "sha256": "766a000da416a8a45760fce75dd387cc2ba3a358f7c58b4813a7b222ddb32471", "urls": ["https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/ec79a486dac00511c26bf3a04531c2c5147e723a/README.md"]}]}]}')

DPC = Path('/opt/dpcvol/datasets')
DEFAULT_ROOT = DPC / 'naie/task-vector/assets/official-v1'
DEFAULT_EXPERIMENT = DPC / 'naie/task-vector/experiments/exp-20261011-01'
DEFAULT_RUNTIME = DPC / 'naie/task-vector/runtime/py28-transformers4513-20261011'
PRINT_LOCK = threading.Lock()
BLOCK = 4 * 1024 * 1024
TLS_CONTEXT = None


def certificate_failure(exc):
    return isinstance(getattr(exc, 'reason', exc), ssl.SSLCertVerificationError)


def curl_base():
    executable = shutil.which('curl')
    if not executable:
        raise RuntimeError('System curl is unavailable and Python HTTPS verification failed')
    # Reuse the verified-HTTPS curl route used by safety's network recovery tool.
    # Arguments remain a list, and proxy/CA settings are inherited from the platform.
    return [executable, '--fail', '--location', '--silent', '--show-error', '--connect-timeout', '25']


def curl_probe(name, url):
    result = subprocess.run(curl_base() + ['--max-time', '30', '--range', '0-0', '--max-filesize', '4096',
                            '--output', os.devnull, '--write-out', '%{http_code}\n%{url_effective}', url],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=40)
    output = result.stdout.splitlines()
    status = int(output[0]) if output and output[0].isdigit() else 0
    # Exit 63 means our deliberate 4KiB guard stopped a server ignoring Range.
    # Its TLS chain and actual GET response were already verified by curl.
    if result.returncode not in (0, 63) or not 200 <= status < 300:
        raise RuntimeError(f'curl probe failed: exit={result.returncode} HTTP={status}: {result.stderr.strip()}')
    say(f'SOURCE_TLS_PASS={name} HTTP={status} transport=curl')
    return {'source': name, 'requested_url': url, 'resolved_url': output[-1],
            'http_status': status, 'transport': 'curl', 'range_guard_exit': result.returncode}


def curl_download(url, partial, spec, attempt):
    header_file = partial.with_name(partial.name + '.http-headers')
    offset = partial.stat().st_size if partial.exists() else 0
    args = curl_base() + ['--speed-limit', '1024', '--speed-time', '90', '--continue-at', '-',
                         '--dump-header', str(header_file), '--output', str(partial),
                         '--write-out', '%{http_code}\n%{url_effective}', url]
    say(f"DOWNLOAD {spec['destination']} attempt={attempt} resume={offset} transport=curl source={url}")
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started, last = time.monotonic(), time.monotonic()
    try:
        while process.poll() is None:
            time.sleep(.5)
            if time.monotonic() - last >= 15:
                size = partial.stat().st_size if partial.exists() else offset
                elapsed = max(time.monotonic() - started, .001)
                say(f"PROGRESS {spec['destination']} {size}/{spec['size']} {100*size/spec['size']:.1f}% "
                    f"{max(size-offset,0)/elapsed/1048576:.2f}MiB/s transport=curl")
                last = time.monotonic()
        output, error = process.communicate()
    except BaseException:
        process.terminate()
        process.communicate()
        raise
    if process.returncode == 33 and partial.exists():
        # curl refuses unsafe append when the server ignores Range. Preserve the
        # owned partial, then let the next outer attempt start from byte zero.
        retain_bad(partial)
        raise OSError('curl source does not support resuming; preserved partial and will restart')
    if process.returncode == 60:
        raise ssl.SSLCertVerificationError('curl certificate verification failed: ' + error.strip())
    if process.returncode:
        raise OSError(f'curl download failed: exit={process.returncode}: {error.strip()}')
    lines = output.splitlines()
    status = int(lines[0]) if lines and lines[0].isdigit() else 0
    if status not in (200, 206):
        raise ValueError(f'Unexpected curl HTTP status: {status}')
    headers = {}
    if header_file.exists():
        for line in header_file.read_text(encoding='iso-8859-1').splitlines():
            if line.startswith('HTTP/'):
                headers = {}
            elif ':' in line:
                key, value = line.split(':', 1)
                headers[key.lower()] = value.strip()
    etag = headers.get('etag', '').strip('"')
    if spec.get('etag') and not spec.get('sha256') and etag != spec['etag']:
        raise ValueError(f'curl release ETag changed or missing: {etag!r}')
    return {'requested_url': url, 'resolved_url': lines[-1], 'etag': etag,
            'last_modified': headers.get('last-modified'), 'http_status': status, 'transport': 'curl'}


def configure_tls():
    """Load existing CA stores explicitly for custom platform Python builds.

    Keep server-chain and hostname verification enabled. No global environment
    changes, package upgrades, or unverified HTTPS fallback are performed.
    """
    global TLS_CONTEXT
    context = ssl.create_default_context()
    evidence = {'openssl': ssl.OPENSSL_VERSION,
                'default_verify_paths': ssl.get_default_verify_paths()._asdict(),
                'default_store': context.cert_store_stats(), 'bundles': [], 'unavailable': []}
    candidates = []
    for variable in ('SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'PIP_CERT'):
        if os.environ.get(variable):
            candidates.append((variable, os.environ[variable], True))
    for name in ('certifi', 'pip._vendor.certifi'):
        try:
            candidates.append((name, importlib.import_module(name).where(), False))
        except (ImportError, AttributeError) as exc:
            evidence['unavailable'].append({'source': name, 'reason': type(exc).__name__})
    for path in ('/etc/ssl/certs/ca-certificates.crt', '/etc/pki/tls/certs/ca-bundle.crt',
                 '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem', '/etc/ssl/ca-bundle.pem'):
        candidates.append(('system_bundle', path, False))
    seen = set()
    for source, raw, explicit in candidates:
        path = Path(raw).expanduser()
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.exists():
            if explicit:
                raise FileNotFoundError(f'Configured CA bundle does not exist: {source}={path}')
            continue
        try:
            if path.is_dir():
                context.load_verify_locations(capath=str(path))
                evidence['bundles'].append({'source': source, 'path': resolved, 'type': 'CA_directory'})
            else:
                context.load_verify_locations(cafile=str(path))
                evidence['bundles'].append({'source': source, 'path': resolved, 'sha256': sha256(path)})
        except (OSError, ssl.SSLError) as exc:
            if explicit:
                raise
            evidence['unavailable'].append({'source': source, 'path': resolved, 'reason': str(exc)})
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise RuntimeError('Internal error: HTTPS verification must remain enabled')
    context.set_alpn_protocols(['http/1.1'])
    evidence['effective_store'] = context.cert_store_stats()
    evidence['hostname_verification'] = True
    evidence['proxy_schemes_present'] = sorted(urllib.request.getproxies())
    TLS_CONTEXT = context
    say('TLS_CA=' + json.dumps(evidence, ensure_ascii=False))
    return evidence


def probe_source(name, url):
    headers = {'User-Agent': 'task-vector-assets/1.1', 'Accept-Encoding': 'identity', 'Range': 'bytes=0-0'}
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=25, context=TLS_CONTEXT)
    except urllib.error.URLError as exc:
        if not certificate_failure(exc) or not shutil.which('curl'):
            raise
        return curl_probe(name, url)
    with response:
        if not 200 <= response.status < 300:
            raise ValueError(f'HTTP {response.status}')
        # Follow the real GET/CDN route; consume only one byte even if Range is ignored.
        response.read(1)
        result = {'source': name, 'requested_url': url, 'resolved_url': response.url,
                  'http_status': response.status, 'etag': response.headers.get('ETag'), 'transport': 'urllib'}
        say(f'SOURCE_TLS_PASS={name} HTTP={response.status}')
        return result


def network_preflight():
    models = MANIFEST['models']
    qwen = next(model for model in models if model['repo_id'] == 'Qwen/Qwen3-1.7B')
    weight = next(item for item in qwen['files'] if item['path'].endswith('.safetensors'))
    dataset = MANIFEST['datasets'][0]['files'][0]
    probes = [('model_mirror', weight['urls']), ('hf-mirror', [dataset['url']])]
    probes.extend((archive['key'], [archive['url']]) for archive in MANIFEST['archives'])

    def probe_routes(name, urls):
        errors = []
        for url in urls:
            try:
                record = probe_source(name, url)
                record['previous_route_errors'] = errors
                return record
            except Exception as exc:
                errors.append({'url': url, 'error': f'{type(exc).__name__}: {exc}'})
        raise RuntimeError(json.dumps(errors, ensure_ascii=False))

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(probe_routes, name, urls): name for name, urls in probes}
        for future in as_completed(futures):
            name = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                record = {'source': name, 'error': f'{type(exc).__name__}: {exc}'}
                records.append(record)
                say('SOURCE_TLS_FAIL=' + json.dumps(record, ensure_ascii=False))
    return records


def say(message):
    with PRINT_LOCK:
        print(message, flush=True)


def utc():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(BLOCK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    os.replace(temporary, path)


def safe_relative(raw):
    relative = PurePosixPath(raw)
    if not raw or relative.is_absolute() or '..' in relative.parts or '\\' in raw or ':' in raw:
        raise ValueError(f'Unsafe archive/manifest path: {raw!r}')
    return relative


def child(root, relative):
    relative = safe_relative(relative)
    target = Path(root).joinpath(*relative.parts)
    resolved = target.resolve()
    if Path(root).resolve() not in resolved.parents:
        raise ValueError(f'Path escapes asset directory: {target}')
    return target


def dpc_directory(raw):
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError('Use an absolute DPC directory')
    resolved = path.resolve()
    if DPC not in resolved.parents:
        raise ValueError(f'Directory must resolve under {DPC}: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return resolved


def file_spec(parent, item, kind):
    item = dict(item)
    item['key'] = parent['key']
    item['kind'] = kind
    if kind == 'model':
        item['destination'] = f"models/{parent['key']}/{item['path']}"
    elif kind == 'dataset':
        item['destination'] = f"data/{parent['key']}/{item['path']}"
        item['sha256'] = item.get('lfs', {}).get('sha256')
        item['urls'] = [item['url']]
    else:
        item['destination'] = f"archives/{item['filename']}"
        item['urls'] = [item['url']]
    safe_relative(item['destination'])
    if not item.get('sha256') and not item.get('etag'):
        raise ValueError(f"No immutable checksum or release ETag: {item['destination']}")
    return item


def all_specs():
    specs = []
    for parent in MANIFEST['models']:
        specs.extend(file_spec(parent, item, 'model') for item in parent['files'])
    for parent in MANIFEST['datasets']:
        specs.extend(file_spec(parent, item, 'dataset') for item in parent['files'])
    for archive in MANIFEST['archives']:
        specs.append(file_spec(archive, archive, 'archive'))
    return specs


def verify_file(path, spec):
    if path.stat().st_size != spec['size']:
        raise ValueError(f"Size mismatch: {path}: {path.stat().st_size} != {spec['size']}")
    actual = sha256(path)
    expected = spec.get('sha256')
    if expected and actual != expected:
        raise ValueError(f'SHA256 mismatch: {path}: {actual} != {expected}')
    return actual


def retain_bad(path):
    """Preserve a corrupt/interrupted download rather than deleting user files."""
    saved = path.with_name(path.name + '.invalid-' + utc())
    os.rename(path, saved)
    say(f'PRESERVED_INVALID={saved}')


def stream_download(url, partial, spec, attempt):
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec['size']:
        retain_bad(partial)
        offset = 0
    headers = {'User-Agent': 'task-vector-assets/1.0', 'Accept-Encoding': 'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
        if spec.get('etag'):
            headers['If-Range'] = '"' + spec['etag'] + '"'
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=90, context=TLS_CONTEXT)
    except urllib.error.URLError as exc:
        if not certificate_failure(exc) or not shutil.which('curl'):
            raise
        return curl_download(url, partial, spec, attempt)
    with response:
        status = response.status
        etag = response.headers.get('ETag', '').strip('"')
        if spec.get('etag') and not spec.get('sha256') and etag != spec['etag']:
            raise ValueError(f'Release ETag changed or missing: {etag!r}')
        if status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if not match or int(match[1]) != offset or int(match[3]) != spec['size']:
                raise ValueError(f"Invalid resume response: {response.headers.get('Content-Range')}")
            mode = 'ab' if offset else 'wb'
        elif status == 200:
            # This path is an owned partial file, never a completed asset.
            mode, offset = 'wb', 0
        else:
            raise ValueError(f'Unexpected HTTP status {status}')
        started, last = time.monotonic(), time.monotonic()
        downloaded = offset
        say(f"DOWNLOAD {spec['destination']} attempt={attempt} resume={offset} source={url}")
        with partial.open(mode) as output:
            while True:
                chunk = response.read(BLOCK)
                if not chunk:
                    break
                if downloaded + len(chunk) > spec['size']:
                    raise ValueError('Server sent more bytes than the pinned file size')
                output.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last >= 15:
                    elapsed = max(time.monotonic() - started, .001)
                    say(f"PROGRESS {spec['destination']} {downloaded}/{spec['size']} "
                        f"{100*downloaded/spec['size']:.1f}% {(downloaded-offset)/elapsed/1048576:.2f}MiB/s")
                    last = time.monotonic()
        return {'requested_url': url, 'resolved_url': response.url, 'etag': etag,
                'last_modified': response.headers.get('Last-Modified'), 'http_status': status}


def acquire(root, spec, verify_only=False):
    target = child(root, spec['destination'])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # A mismatch is never overwritten silently.
        digest = verify_file(target, spec)
        origin_file = target.with_name(target.name + '.origin.json')
        origin = json.loads(origin_file.read_text(encoding='utf-8')) if origin_file.exists() else {}
        if not spec.get('sha256') and origin.get('sha256') not in (None, digest):
            raise ValueError(f'Original-release asset changed after first download: {target}')
        say(f"VERIFIED_EXISTING={spec['destination']}")
        return {'path': str(target), 'sha256': digest, 'size': target.stat().st_size, 'reused': True,
                'origin': origin}
    if verify_only:
        raise FileNotFoundError(f'Missing local asset: {target}')
    partial = target.with_name(target.name + '.partial')
    errors = []
    for url in spec['urls']:
        for attempt in range(1, 4):
            try:
                origin = {}
                if not partial.exists() or partial.stat().st_size != spec['size']:
                    origin = stream_download(url, partial, spec, attempt)
                try:
                    digest = verify_file(partial, spec)
                except ValueError:
                    if partial.exists() and partial.stat().st_size == spec['size']:
                        retain_bad(partial)
                    raise
                # No concurrent second writer: main() holds an OS file lock.
                if target.exists():
                    raise FileExistsError(target)
                os.rename(partial, target)
                origin.update(sha256=digest, size=spec['size'], completed_utc=utc(),
                              expected_sha256=spec.get('sha256'), expected_etag=spec.get('etag'),
                              requested_url=origin.get('requested_url', url))
                write_json(target.with_name(target.name + '.origin.json'), origin)
                say(f"DOWNLOAD_OK={spec['destination']}")
                return {'path': str(target), 'sha256': digest, 'size': spec['size'], 'origin': origin,
                        'reused': False}
            except (OSError, ValueError, http.client.HTTPException) as exc:
                message = f'{type(exc).__name__}: {exc}'
                errors.append({'url': url, 'attempt': attempt, 'error': message})
                say(f"DOWNLOAD_RETRY {spec['destination']}: {message}")
                reason = getattr(exc, 'reason', exc)
                if isinstance(reason, ssl.SSLCertVerificationError):
                    # Certificate failures are deterministic; try the next pinned source.
                    break
                if attempt < 3:
                    time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(json.dumps(errors, ensure_ascii=False))


def extract_proofwriter(root, archive):
    extracted = child(root, 'data/proofwriter')
    extracted.mkdir(parents=True, exist_ok=True)
    records, split_paths = [], defaultdict(list)
    with zipfile.ZipFile(child(root, 'archives/' + archive['filename'])) as source:
        for item in archive['members']:
            info = source.getinfo(item['path'])
            if stat.S_ISLNK(info.external_attr >> 16) or info.file_size != item['size']:
                raise ValueError(f'Unexpected ProofWriter archive member: {info.filename}')
            target = child(extracted, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Streaming ZIP read checks the CRC of each selected original member.
            digest = hashlib.sha256()
            temporary = target.with_name(target.name + '.extracting')
            writer = None if target.exists() else temporary.open('wb')
            try:
                with source.open(info) as original:
                    for block in iter(lambda: original.read(BLOCK), b''):
                        digest.update(block)
                        if writer:
                            writer.write(block)
            finally:
                if writer:
                    writer.close()
            expected = digest.hexdigest()
            if target.exists():
                if target.stat().st_size != info.file_size or sha256(target) != expected:
                    raise ValueError(f'Existing extracted file differs from original: {target}')
            else:
                os.rename(temporary, target)
            theories, queries, labels, question_depths = 0, 0, Counter(), Counter()
            with target.open(encoding='utf-8') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    theories += 1
                    if not row.get('theory') and not (row.get('triples') or row.get('rules')):
                        raise ValueError(f'Missing original ProofWriter theory: {target}')
                    questions = row.get('questions')
                    if not isinstance(questions, dict) or not questions:
                        raise ValueError(f'Missing original ProofWriter question map: {target}')
                    for question in questions.values():
                        label = str(question['answer']).capitalize()
                        if label not in {'True', 'False', 'Unknown'} or not isinstance(question.get('question'), str):
                            raise ValueError(f'Invalid original OWA query: {target}')
                        queries += 1
                        labels[label] += 1
                        question_depths[str(question.get('QDep'))] += 1
            split = 'validation' if item['split'] == 'dev' else item['split']
            split_paths[split].append(str(target))
            records.append({'path': str(target), 'sha256': expected, 'size': info.file_size,
                            'split': split, 'source_depth': item['depth'], 'theories': theories,
                            'queries': queries, 'labels': dict(labels), 'QDep': dict(question_depths),
                            'verification': 'original ZIP member CRC and complete streamed SHA256'})
    if set(split_paths) != {'train', 'validation', 'test'} or any(len(v) != 4 for v in split_paths.values()):
        raise ValueError('Expected exact OWA depths 0..3, three official splits')
    labels = Counter()
    for row in records:
        labels.update(row['labels'])
    if set(labels) != {'True', 'False', 'Unknown'}:
        raise ValueError('OWA must retain all three original labels')
    return dict(split_paths), records


def extract_samsum(root, archive):
    import py7zr
    extracted = child(root, 'data/samsum')
    extracted.mkdir(parents=True, exist_ok=True)
    source_path = child(root, 'archives/' + archive['filename'])
    with py7zr.SevenZipFile(source_path, 'r') as source:
        entries = source.list()
        for info in entries:
            safe_relative(info.filename)
            if getattr(info, 'is_symlink', False):
                raise ValueError('SAMSum archive may not contain links')
        names = [info.filename for info in entries if not info.is_directory]
    # Extract to a unique staging folder. Keep it as the canonical original copy,
    # avoiding any replacement of an earlier extraction or user-provided file.
    staging = extracted / ('original-' + archive['sha256'][:16])
    complete = staging / '.extraction-complete.json'
    receipt = json.loads(complete.read_text(encoding='utf-8')) if complete.exists() else {}
    if receipt and (receipt.get('archive_sha256') != archive['sha256'] or not receipt.get('member_hashes')):
        staging = extracted / ('original-' + archive['sha256'][:16] + '-' + utc())
        complete = staging / '.extraction-complete.json'
    if not complete.exists():
        if staging.exists() and any(staging.iterdir()):
            staging = extracted / ('original-' + archive['sha256'][:16] + '-' + utc())
            complete = staging / '.extraction-complete.json'
        staging.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(source_path, 'r') as source:
            source.extractall(path=staging)
        # The archive has an expected SHA256; CRC errors also abort extraction.
        member_hashes = {name: {'size': child(staging, name).stat().st_size,
                                'sha256': sha256(child(staging, name))} for name in names}
        write_json(complete, {'archive_sha256': archive['sha256'], 'members': names,
                              'member_hashes': member_hashes})
    receipt = json.loads(complete.read_text(encoding='utf-8'))
    if receipt['archive_sha256'] != archive['sha256'] or set(receipt['member_hashes']) != set(names):
        raise ValueError('SAMSum extraction receipt does not match the pinned original archive')
    for name, evidence in receipt['member_hashes'].items():
        path = child(staging, name)
        if path.stat().st_size != evidence['size'] or sha256(path) != evidence['sha256']:
            raise ValueError(f'Extracted SAMSum file changed after extraction: {path}')
    for name, expected_hash in archive.get('member_sha256', {}).items():
        if sha256(child(staging, name)) != expected_hash:
            raise ValueError(f'SAMSum original member SHA256 mismatch: {name}')
    splits, records = {}, []
    for split, aliases in {'train': {'train'}, 'validation': {'val', 'valid', 'validation'}, 'test': {'test'}}.items():
        matches = [name for name in names if PurePosixPath(name).suffix.lower() == '.json'
                   and PurePosixPath(name).stem.lower() in aliases]
        if len(matches) != 1:
            raise ValueError(f'SAMSum {split}: expected exactly one original JSON file; found {matches}')
        path = child(staging, matches[0])
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(rows, list) or len(rows) != archive['expected_rows'][split]:
            raise ValueError(f'SAMSum {split}: official row count mismatch')
        ids = []
        for row in rows:
            if not all(isinstance(row.get(key), str) for key in ('id', 'dialogue', 'summary')):
                raise ValueError(f'SAMSum {split}: original fields missing')
            ids.append(row['id'])
        if len(set(ids)) != len(ids):
            raise ValueError(f'SAMSum {split}: duplicate original IDs')
        splits[split] = str(path)
        records.append({'path': str(path), 'split': split, 'rows': len(rows), 'sha256': sha256(path)})
    return splits, records


def validate_datasets(root):
    import pyarrow.parquet as pq
    sources, validations = {}, []
    required = {'gsm8k': {'question', 'answer'}, 'qqp': {'question1', 'question2', 'label', 'idx'},
                'mbpp': {'task_id', 'text', 'code', 'test_list', 'test_setup_code', 'challenge_test_list'},
                'iwslt2017': {'translation'}}
    for dataset in MANIFEST['datasets']:
        key, files = dataset['key'], {}
        for spec in dataset['files']:
            path = child(root, f"data/{key}/{spec['path']}")
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != spec['expected_rows']:
                raise ValueError(f'Official row count mismatch: {path}')
            if not required[key].issubset(parquet.schema_arrow.names):
                raise ValueError(f'Official schema mismatch: {path}: {parquet.schema_arrow.names}')
            if key == 'mbpp':
                ids = parquet.read(columns=['task_id']).column('task_id').to_pylist()
                lo, hi = dataset['task_id_ranges'][spec['split']]
                if sorted(ids) != list(range(lo, hi + 1)):
                    raise ValueError(f'Full MBPP official program IDs mismatch: {path}')
            elif key == 'iwslt2017':
                field = parquet.schema_arrow.field('translation')
                if set(field.type.names) != {'en', 'de'}:
                    raise ValueError(f'IWSLT must contain original en/de translations: {path}')
            elif key == 'gsm8k':
                if any('####' not in text for text in parquet.read(columns=['answer']).column('answer').to_pylist()):
                    raise ValueError('GSM8K must retain complete original answers')
            elif key == 'qqp':
                counts = Counter(parquet.read(columns=['label']).column('label').to_pylist())
                expected = {-1} if spec['split'] == 'test' else {0, 1}
                if set(counts) != expected:
                    raise ValueError(f'Unexpected official QQP labels: {counts}')
            files[spec['split']] = str(path)
            validations.append({'dataset': key, 'split': spec['split'], 'path': str(path),
                                'rows': parquet.metadata.num_rows, 'schema': str(parquet.schema_arrow)})
        sources[key] = {'files': files, 'dataset_id': dataset['repo_id'],
                        'config': dataset['config'], 'revision': dataset['revision'], 'allow_remote': False}
    return sources, validations


def validate_models(root):
    models = {model['key']: model for model in MANIFEST['models']}
    qwen = next(key for key, model in models.items() if model['repo_id'] == 'Qwen/Qwen3-1.7B')
    mini = next(key for key, model in models.items() if model['repo_id'] == 'sentence-transformers/all-MiniLM-L6-v2')
    qwen_path, mini_path = child(root, 'models/' + qwen), child(root, 'models/' + mini)
    config = json.loads((qwen_path / 'config.json').read_text())
    expected = {'model_type': 'qwen3', 'num_hidden_layers': 28, 'hidden_size': 2048,
                'num_attention_heads': 16, 'num_key_value_heads': 8, 'head_dim': 128}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Qwen configuration disagrees with experiment: {config}')
    index = json.loads((qwen_path / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    if not shards or any(not child(qwen_path, shard).is_file() for shard in shards):
        raise ValueError('Qwen weight index references missing shards')
    modules = json.loads((mini_path / 'modules.json').read_text())
    types = [module['type'] for module in modules]
    if not any(kind.endswith('.Pooling') for kind in types) or not any(kind.endswith('.Normalize') for kind in types):
        raise ValueError('MiniLM must retain its original pooling and normalization modules')
    if not (mini_path / '1_Pooling/config.json').is_file():
        raise ValueError('Missing MiniLM pooling configuration')
    return qwen_path, mini_path, {'qwen_config': expected, 'qwen_shards': sorted(shards), 'minilm_modules': types}


def runtime_contract(runtime):
    if Path(sys.prefix).resolve() != runtime.resolve():
        raise ValueError(f'Use private experiment Python: {runtime}/bin/python; got {sys.executable}')
    versions = {name: metadata.version(name) for name in ('torch', 'torch-npu', 'transformers', 'numpy', 'datasets', 'py7zr', 'pyarrow')}
    expected = {'torch': '2.8.0', 'torch-npu': '2.8.0.post5', 'transformers': '4.51.3',
                'numpy': '1.26.4', 'datasets': '3.6.0'}
    if any(versions[key].split('+')[0] != value for key, value in expected.items()):
        raise ValueError(f'Protected runtime versions changed: {versions}')
    import py7zr  # Check extraction dependencies before any multi-GB download.
    import pyarrow.parquet
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', default=str(DEFAULT_ROOT))
    parser.add_argument('--experiment-root', default=str(DEFAULT_EXPERIMENT))
    parser.add_argument('--runtime', default=str(DEFAULT_RUNTIME))
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--network-check-only', action='store_true', help='Check CA trust and required source endpoints without downloading assets')
    parser.add_argument('--plan', action='store_true', help='Print pinned download sizes without imports, writes or network')
    args = parser.parse_args(argv)
    specs = all_specs()
    manifest_digest = hashlib.sha256(json.dumps(MANIFEST, sort_keys=True).encode()).hexdigest()
    total = sum(spec['size'] for spec in specs)
    if args.plan:
        print(json.dumps({'files': len(specs), 'download_bytes': total, 'download_GiB': total / 2**30,
                          'manifest_sha256': manifest_digest, 'models': [m['repo_id'] for m in MANIFEST['models']],
                          'datasets': [d['repo_id'] for d in MANIFEST['datasets']],
                          'original_archives': [a['source_identity'] for a in MANIFEST['archives']]}, indent=2))
        return 0
    report, lock = {'started_utc': utc(), 'manifest_sha256': manifest_digest, 'downloads': [], 'errors': []}, None
    report_path = None
    try:
        runtime = Path(args.runtime)
        report['runtime_versions'] = runtime_contract(runtime)
        root, experiment = dpc_directory(args.assets_root), dpc_directory(args.experiment_root)
        import fcntl
        lock = (root / '.download.lock').open('a+')
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another asset downloader is active in the same directory; wait for its result')
        report_path = experiment / 'environment' / f"assets-{report['started_utc']}.json"
        report['assets_root'] = str(root)
        report['script_sha256'] = sha256(Path(__file__))
        report['source_manifest'] = MANIFEST
        say(f'DOWNLOAD_BYTES={total} ({total / 2**30:.3f} GiB); files={len(specs)}; workers={args.workers}')
        say(f'ASSETS_ROOT={root}')
        write_json(report_path, report)
        if not args.verify_only:
            report['tls'] = configure_tls()
            report['source_preflight'] = network_preflight()
            write_json(report_path, report)
            failures = [record for record in report['source_preflight'] if record.get('error')]
            if failures:
                raise RuntimeError('Source connection preflight failed; no asset downloads started. '
                                   'See SOURCE_TLS_FAIL and the saved TLS evidence.')
            say('NETWORK_READY=PASS')
            if args.network_check_only:
                report.update(status='NETWORK_PASS', completed_utc=utc())
                write_json(report_path, report)
                say(f'ASSETS_RECEIPT={report_path}')
                return 0
            model_route = next(record for record in report['source_preflight'] if record['source'] == 'model_mirror')
            preferred_host = urllib.parse.urlsplit(model_route['requested_url']).netloc
            for spec in specs:
                if spec['kind'] == 'model':
                    spec['urls'] = sorted(spec['urls'], key=lambda url: urllib.parse.urlsplit(url).netloc != preferred_host)
        # Largest files start first. Downloads are independent; a small-file failure
        # is reported after other assets finish, so a rerun can reuse completed work.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(acquire, root, spec, args.verify_only): spec
                       for spec in sorted(specs, key=lambda item: item['size'], reverse=True)}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    record = future.result()
                    record.update(key=spec['key'], kind=spec['kind'], destination=spec['destination'])
                    report['downloads'].append(record)
                except Exception as exc:
                    error = {'destination': spec['destination'], 'error': f'{type(exc).__name__}: {exc}'}
                    report['errors'].append(error)
                    say(f'ASSET_FAILED={json.dumps(error, ensure_ascii=False)}')
                write_json(report_path, report)
        if report['errors']:
            raise RuntimeError(f"{len(report['errors'])} asset files failed; rerun the same command to resume")
        sources, report['dataset_validation'] = validate_datasets(root)
        archives = {archive['key']: archive for archive in MANIFEST['archives']}
        say('VALIDATING_PROOFWRITER=OWA depth 0..3 canonical meta files (streaming original records)')
        logic, report['proofwriter_validation'] = extract_proofwriter(root, archives['proofwriter'])
        say('VALIDATING_SAMSUM=original train/val/test')
        summary, report['samsum_validation'] = extract_samsum(root, archives['samsum'])
        qwen, mini, report['model_validation'] = validate_models(root)
        data_sources = {'math': sources['gsm8k'], 'translation': sources['iwslt2017'], 'paraphrase': sources['qqp'],
                        'code_generation': sources['mbpp'], 'code_repair': sources['mbpp'],
                        'logic': {'files': logic, 'world': 'OWA', 'max_depth': 3, 'revision': 'V2020.12.3', 'allow_remote': False},
                        'summary': {'files': summary, 'dataset_id': 'Samsung/samsum', 'revision': 'arxiv-1911.12237v2', 'allow_remote': False}}
        paths = {'model_path': str(qwen), 'semantic_model_path': str(mini), 'persistent_root': str(experiment),
                 'runtime_python': str(runtime / 'bin/python'), 'cann_env': str(runtime / 'activate-task-vector.sh'),
                 'expected_devices': 8, 'data_sources': data_sources}
        for name in ('runtime_python', 'cann_env'):
            if not Path(paths[name]).is_file():
                raise FileNotFoundError(paths[name])
        # A new receipt/config path per attempt keeps earlier records reviewable.
        paths_file = experiment / 'environment' / f"asset-paths-{report['started_utc']}.json"
        write_json(paths_file, paths)
        report.update(status='PASS', completed_utc=utc(), paths_json=str(paths_file))
        write_json(report_path, report)
        say('ASSETS_READY=PASS')
        say(f'PATHS_JSON={paths_file}')
        say(f'ASSETS_RECEIPT={report_path}')
        say('NEXT=configure_local and prepare/preflight; no training job has been submitted')
        return 0
    except Exception as exc:
        report.update(status='FAIL', completed_utc=utc(), failure=f'{type(exc).__name__}: {exc}')
        if report_path:
            write_json(report_path, report)
        say('ASSETS_READY=FAIL')
        say(report['failure'])
        if report_path:
            say(f'ASSETS_RECEIPT={report_path}')
        return 1
    finally:
        if lock:
            lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
