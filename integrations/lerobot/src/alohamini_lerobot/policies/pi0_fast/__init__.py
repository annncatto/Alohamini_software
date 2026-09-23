#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from alohamini_lerobot.policies import _load_export

_EXPORTS = {
    "PI0FastConfig": (".configuration_pi0_fast", "PI0FastConfig"),
    "PI0FastPolicy": (".modeling_pi0_fast", "PI0FastPolicy"),
    "make_pi0_fast_pre_post_processors": (
        ".processor_pi0_fast",
        "make_pi0_fast_pre_post_processors",
    ),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _load_export(__name__, _EXPORTS, name)
