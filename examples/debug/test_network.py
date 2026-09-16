# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test explicitly supplied HTTP(S) endpoints, without network discovery."""

import argparse
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "urls", nargs="+", help="HTTP(S) URLs to test; no default external services"
    )
    args = parser.parse_args(argv)
    if any(urlsplit(url).scheme not in ("http", "https") for url in args.urls):
        parser.error("Only explicit HTTP(S) URLs are supported")
    success = True
    for url in args.urls:
        try:
            with urlopen(url, timeout=5.0) as response:
                print("Connection successful:", url, response.status)
        except (URLError, TimeoutError, OSError) as exc:
            print("Connection failed:", url, exc)
            success = False
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
