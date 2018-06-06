#!/usr/bin/env python
from os import environ, path

from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

MODELDIR = "pocketsphinx/model"
DATADIR = "pocketsphinx/test/data"

# Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', path.join(MODELDIR, 'en-us/en-us'))
#config.set_string('-lm', path.join(MODELDIR, 'en-us/en-us.lm.bin'))
#config.set_string('-dict', path.join(MODELDIR, 'en-us/cmudict-en-us.dict'))
config.set_string('-allphone', path.join(MODELDIR, 'en-us/en-us-phone.lm.bin'))
#config.set_string('-backtrace', 'yes')
#config.set_string('-beam', 1e-20)
#config.set_string('-pbeam', 1e-20)
#config.set_string('-lw', '2.0')

decoder = Decoder(config)

# Decode streaming data.
decoder = Decoder(config)
decoder.start_utt()
stream = open(path.join(DATADIR, 'farai-mono.wav'), 'r')
while True:
  buf = stream.read(1024)
  if buf:
    decoder.process_raw(buf, False, False)
  else:
    break
decoder.end_utt()
print ('Best hypothesis segments: ', [seg.word for seg in decoder.seg()])
