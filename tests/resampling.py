import librosa

FILENAME = "/Users/quentin/Downloads/test.wav"

y, sr = librosa.load(FILENAME, sr=22050, mono=False)
librosa.output.write_wav('/Users/quentin/Downloads/22050.wav', y, sr)

y, sr = librosa.load(FILENAME, sr=44100, mono=False)
librosa.output.write_wav('/Users/quentin/Downloads/44100.wav', y, sr)

y, sr = librosa.load(FILENAME, sr=8000, mono=False)
librosa.output.write_wav('/Users/quentin/Downloads/8000.wav', y, sr)

y, sr = librosa.load(FILENAME, sr=16000, mono=False)
librosa.output.write_wav('/Users/quentin/Downloads/16000.wav', y, sr)
