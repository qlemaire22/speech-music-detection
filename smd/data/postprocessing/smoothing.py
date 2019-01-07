from smd import config


def smooth_output(output, min_speech=1.3, min_music=3.4, max_silence_speech=0.4, max_silence_music=0.6):
    duration_frame = config.HOP_LENGTH / config.SAMPLING_RATE
    n_frame = output.shape[1]

    start_music = -1000
    start_speech = -1000

    for i in range(n_frame):
        if output[0, i] == 1:
            if i - start_speech > 1:
                if (i - start_speech) * duration_frame <= max_silence_speech:
                    output[0, start_speech:i] = 1

            start_speech = i

        if output[1, i] == 1:
            if i - start_music > 1:
                if (i - start_music) * duration_frame <= max_silence_music:
                    output[1, start_music:i] = 1

            start_music = i

    start_music = -1000
    start_speech = -1000

    for i in range(n_frame):
        if i != n_frame - 1:
            if output[0, i] == 0:
                if i - start_speech > 1:
                    if (i - start_speech) * duration_frame <= min_speech:
                        output[0, start_speech:i] = 0

                start_speech = i

            if output[1, i] == 0:
                if i - start_music > 1:
                    if (i - start_music) * duration_frame <= min_music:
                        output[1, start_music:i] = 0

                start_music = i
        else:
            if i - start_speech > 1:
                if (i - start_speech) * duration_frame <= min_speech:
                    output[0, start_speech:i + 1] = 0

            if i - start_music > 1:
                if (i - start_music) * duration_frame <= min_music:
                    output[1, start_music:i + 1] = 0

    return output
