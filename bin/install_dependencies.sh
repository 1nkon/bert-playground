#!/bin/bash

abort() {
  echo "$1"
  exit 1
}

[ -x "$(command -v pip)" ] || abort "pip not found"
[ -x "$(command -v wget)" ] || abort "wget not found"
[ -x "$(command -v gunzip)" ] || abort "gunzip not found"

[ ! -f data/cc.en.300.bin.gz ] && \
  [ ! -f data/cc.en.300.bin ] && \
  { wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz -P data || \
  abort "Failed to download fast text data"; }
[ ! -f data/cc.en.300.bin ] && { gunzip data/cc.en.300.bin.gz || abort "Failed to extract files"; }

[ ! -d /tmp/fastText/ ] && git clone https://github.com/facebookresearch/fastText.git /tmp/fastText
pip install /tmp/fastText || abort "Failed to install fast text python module"
pip install -r bin/py_requirements || abort "Failed to install pip requirements from bin/py_requirements"

rm -rf /tmp/fastText