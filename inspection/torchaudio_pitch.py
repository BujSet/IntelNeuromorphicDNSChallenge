import torch
import torchaudio

SPEECH_FILE = "./training_set/clean/clean_fileid_0.wav"

audio, sample_rate = torchaudio.load(SPEECH_FILE)
print(audio)
print("audio dims " + str(audio.size()))
p1d = (1000,1000)
audio = torch.nn.functional.pad(audio, p1d, "constant", 0)
print(audio)
print("audio dims " + str(audio.size()))


pitch_floor = 50
pitch_ceil = 1000

pitch = torchaudio.functional.detect_pitch_frequency(audio, sample_rate, frame_time=0.008, freq_low=50, freq_high=1000)
print(pitch)
print(pitch.size())
