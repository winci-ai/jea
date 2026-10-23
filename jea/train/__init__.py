# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .ssl_meta_arch import SSLMetaArch


def get_args_parser(*args, **kwargs):
    from .train import get_args_parser as parser
    return parser(*args, **kwargs)


def main(*args, **kwargs):
    from .train import main as train_main
    return train_main(*args, **kwargs)
