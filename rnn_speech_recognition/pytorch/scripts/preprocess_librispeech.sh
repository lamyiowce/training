# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#!/usr/bin/env bash

export BASE=/training-data-speech/LibriSpeech/

python ./utils/convert_librispeech.py --no_wav \
    --input_dir $BASE/train-clean-100 \
    --dest_dir $BASE/wav/train-clean-100 \
    --output_json $BASE/librispeech-train-clean-100-wav.json
python ./utils/convert_librispeech.py --no_wav \
    --input_dir $BASE/train-clean-360 \
    --dest_dir $BASE/wav/train-clean-360 \
    --output_json $BASE/librispeech-train-clean-360-wav.json
python ./utils/convert_librispeech.py --no_wav \
    --input_dir $BASE/train-other-500 \
    --dest_dir $BASE/wav/train-other-500 \
    --output_json $BASE/librispeech-train-other-500-wav.json


python ./utils/convert_librispeech.py \
    --input_dir $BASE/dev-clean \
    --dest_dir $BASE/wav/dev-clean \
    --output_json $BASE/librispeech-dev-clean-wav.json --no_wav
python ./utils/convert_librispeech.py \
    --input_dir $BASE/dev-other \
    --dest_dir $BASE/wav/dev-other \
    --output_json $BASE/librispeech-dev-other-wav.json


python ./utils/convert_librispeech.py \
    --input_dir $BASE/test-clean \
    --dest_dir $BASE/wav/test-clean \
    --output_json $BASE/librispeech-test-clean-wav.json
python ./utils/convert_librispeech.py \
    --input_dir $BASE/test-other \
    --dest_dir $BASE/wav/test-other \
    --output_json $BASE/librispeech-test-other-wav.json

bash scripts/create_sentencepieces.sh
