# tts/patch_coqui_cleaners.py
import TTS.tts.utils.text.cleaners as coqui_cleaners
from tts.text_cleaners import nepali_cleaners

# Inject your cleaner into Coqui's cleaners module
coqui_cleaners.nepali_cleaners = nepali_cleaners