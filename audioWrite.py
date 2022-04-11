import soundfile as sf


def audioWrite(path, x, Fs):
    """Writes an audio file to a specified location (path)

    Input parameters:
        x: array of audio samples to be written
        Fs: sample rate [Hz]"""
    sf.write(path, x, Fs)
    return None
