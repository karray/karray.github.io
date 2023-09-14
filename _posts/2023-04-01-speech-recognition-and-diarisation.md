---
layout: post
title:  Hands-on Guide to Multi-Language Speech Recognition and Speaker diarization
date:   2023-03-31 17:45:00 +0200
tags: Whisper Pyannote Speech-Recognition Speaker-diarization
description: Learn how to transcribe any video in one of 99 languages, identify speakers, and translate text into any of these languages.
---

Multi-Language speech recognition and speaker diarization are two important tasks in the field of audio processing. Speech recognition can be defined as the process of converting spoken language into written text, while speaker diarization involves segmenting an audio recording and assigning each segment to a particular speaker. These techniques are used in a variety of applications, including podcasts and conference transcription.

In this blog post, you will learn how to build a pipeline for multi-language speech recognition and speaker diarization using existing libraries.

<!--more-->

## Introduction

Podcasts are a great example of how this technology can be useful. Podcasts have gained growing popularity, which has led to an increasing demand for tools capable of automatically transcribing and segmenting podcast episodes, thus saving a significant amount of time on manual work. Many podcasts are recorded with multiple speakers and are often distributed in audio format only. By using speaker diarization, podcast producers can automatically identify each speaker and generate subtitles for each one. This not only makes the podcast more accessible to hard-of-hearing listeners but also makes it easier to search for specific topics within the podcast or create chapters for YouTube.

Before diving into the Jupyter notebook, let me briefly introduce three libraries that form the backbone of this pipeline.

[**Denoiser**](https://github.com/facebookresearch/denoiser) is a PyTorch implementation of Meta’s paper [Real Time Speech Enhancement in the Waveform Domain](https://arxiv.org/abs/2006.12847). It is used to remove noise from the background and can enhance speech from the raw waveform in real-time on a laptop CPU.

[**Pyannote**](https://github.com/pyannote/pyannote-audio) is an open-source toolkit for audio segmentation. It can identify and separate speakers.

[**Whisper**](https://github.com/openai/whisper) is OpenAI’s Automatic Speech Recognition system trained on 680,000 hours of multilingual and multitasking data collected from the internet. The researchers show that using such a large and diverse data set can improve tolerance to accents and background noise. It can not only automatically recognize language and speech, but can also translate text into one of 99 languages.

Interestingly, the official announcement for translation into any language has not been made. I accidentally stumbled upon this possibility while experimenting with the model. The repository only says that it can translate one of the languages into English.

This demo also contains an HTML5 video player with custom controls. Specifically, it implements a YouTube-like timeline that is divided into chapters for each speaker.

The Jupyter Notebook can be found on my [GitHub](https://github.com/karray/speech-recognition-and-diarization) or you can run it on [Google Colab](https://colab.research.google.com/github/karray/speech-recognition-and-diarization/blob/main/diar_speech.ipynb). If you encounter any problems, you are welcome to open an issue on GitHub.

## Implementation

The notebook has a Setup section that installs packages and defines helper functions. We will go through all sections and look at each cell step by step.

### Install dependencies

To begin, it's important to install dependencies, and it must be done in a specific order due to conflicts between Pyannote and PyTorch Lightning.

```python
!pip install pyannote.audio==2.1.1 denoiser==0.1.5 moviepy==1.0.3 pydub==0.25.1 git+https://github.com/openai/whisper.git@v20230124
!pip install omegaconf==2.3.0 pytorch-lightning==1.8.4
```

If you want to try out this demo on your own computer, you will need to install [ffmpeg](https://ffmpeg.org/) package since we will process video and audio files.

### Start web server

This cell installs and starts a web server on the Google Colab virtual machine. This is needed to host the HTML player and resources.  

```python
!npm install http-server -g

import subprocess
subprocess.Popen(['http-server', '-p', '8000']);
```

Although Python's built-in [http.server](https://docs.python.org/3/library/http.server.html) could be used, it lacks support for the [Range request](https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests), which is needed to rewind video.

### HTML player template

The next cell defines the HTML5 video player template and contains only a string with JavaScript and CSS.

### Main code

This section contains the most important part of the demo. Let's examine the code more closely. We'll start by importing the required libraries and loading the pretrained models.

```python
# Imports...

denoise_model = pretrained.get_model(Namespace(model_path=None, dns48=False, dns64=False, master64=False, valentini_nc=False)).to(device)
denoise_model.eval()
whisper_model = whisper.load_model("large").to(device)
whisper_model.eval()
```

The `split_audio` function extracts the audio from a video file and divides it into smaller pieces using the MoviePy package, which is a wrapper around `ffmpeg`. This is done to ensure that the audio can fit into the available memory. `chunk_size` controls the duration of the chunks. The function returns the total duration of the video (which is required for building a timeline) and saves the audio chunks into the `tmpdirname` directory for further processing.

```python
def split_audio(tmpdirname, video, chunk_size=120):
    """
    Split audio into chunks of chunk_size
    """
    path = opj(tmpdirname, 'noisy_chunks')
    os.makedirs(path)
    # extract audio from video
    audio = AudioFileClip(video.name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as audio_fp:
        audio.write_audiofile(audio_fp.name, verbose=False)

        # round duration to the next whole integer
        for i, chunk in enumerate(np.arange(0, audio.duration, chunk_size)):
            ffmpeg_extract_subclip(audio_fp.name, chunk, min(chunk + chunk_size, audio.duration),
                                targetname=opj(path, f'{i:09}.wav'))
    return audio.duration
```

The `get_speakers` function removes noise from the chunks, reassembles them back to a cleaned audio file, and passes this file into the Pyannote pipeline for speaker diarization.

```python
def get_speakers(tmpdirname, use_auth_token=True):
    files = find_audio_files(opj(tmpdirname, 'noisy_chunks'))
    dset = Audioset(files, with_path=True,
                    sample_rate=denoise_model.sample_rate, channels=denoise_model.chin, convert=True)
    
    loader = distrib.loader(dset, batch_size=1)
    distrib.barrier()

    print('removing noise...')
    enhanced_chunks = []
    with tempfile.TemporaryDirectory() as denoised_tmpdirname:
        for data in loader:
            noisy_signals, filenames = data
            noisy_signals = noisy_signals.to(device)
            
            with torch.no_grad():
                wav = denoise_model(noisy_signals).squeeze(0)
            wav = wav / max(wav.abs().max().item(), 1)

            name = opj(denoised_tmpdirname, filenames[0].split('/')[-1])
            torchaudio.save(name, wav.cpu(), denoise_model.sample_rate)
            enhanced_chunks.append(name)

        print('reassembling chunks...')
        clips = [AudioFileClip(c) for c in sorted(enhanced_chunks)]
        final_clip = concatenate_audioclips(clips)
        cleaned_path = opj(tmpdirname, 'cleaned.wav')
        final_clip.write_audiofile(cleaned_path, verbose=False)

        print('identifying speakers...')
        # load pre-trained model
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=use_auth_token)
    
        return str(pipeline({'uri': '', 'audio': cleaned_path})).split('\n'), cleaned_path
```

The function returns an array of time codes for the speaker turns and the path to clean audio that will be used for transcription.

As we will be downloading pretrained models from Hugging Face, we need to set `use_auth_token`. We will use [notebook_login](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/login#huggingface_hub.notebook_login) to store the token in the config file. Setting `True`
 indicates that the token will be read from that config file. It is also required to accept Pyannote’s [speaker-diarization](https://huggingface.co/pyannote/speaker-diarization) and [segmentation](https://huggingface.co/pyannote/segmentation) user conditions.

Finally, the function `get_subtitles` transcribes audio and composes a dictionary with subtitles.

```python
def get_subtitles(timecodes, clened_audio_path, language=None):
    if(device == 'cpu'):
        options = whisper.DecodingOptions(language=language, fp16=False)
    else:
        options = whisper.DecodingOptions(language=language)

    timeline = {}
    prev_speaker = None
    prev_start = 0
    for line in timecodes:
        start, end = re.findall(r'\d{2}:\d{2}:\d{2}.\d{3}', line)
        start = str_to_seconds(start)
        end = str_to_seconds(end)
        speaker = re.findall(r'\w+$', line)[0]

        # extract a segment of the audio for a speaker
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as audio_fp:
            ffmpeg_extract_subclip(clened_audio_path, start, end,
                                    targetname=audio_fp.name)

            # load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio(audio_fp.name)
            audio = whisper.pad_or_trim(audio)  
            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
            # decode the audio
            result = whisper.decode(whisper_model, mel, options)

            if(speaker == prev_speaker):
                timeline[prev_start]['text'] += f' <{seconds_to_str(start)}>{result.text}'
                timeline[prev_start]['end'] = end
            else:
                timeline[start] = { 'end': end, 
                                    'speaker': speaker,
                                    'text': f'<v.{speaker}>{speaker}</v>: {result.text}'}
                prev_start = start

            prev_speaker = speaker

    return timeline
```

This function performs speech recognition on the audio segments corresponding to each speaker turn using the pretrained Whisper model. It does this by iterating through the time codes produced by the `get_speakers` function and extracting a segment for each speaker from the clean audio. It then computes the [log-Mel spectrogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) of the audio and passes it into the `whisper_model` function for speech recognition. Finally, the resulting transcription in [VTT](https://developer.mozilla.org/en-US/docs/Web/API/WebVTT_API) format and the speaker's ID are added to the `timeline` dictionary, with the start time of the speaker's turn serving as the `key` for the dictionary.

### UI code

This section defines an input form that allows users to provide a link to a video or upload a file. The form also features a language drop-down menu `Translate to`, which includes all the languages that Whisper supports. If the user selects the `Original` option, the model will receive `None` as the language parameter, indicating that the model should automatically detect the language.

Additionally, there are helper functions provided for displaying the video player directly in the notebook or in a new tab. Depending on the situation we need to replace URL placeholders for the video and subtitles.

If the video player is rendered within the notebook, the base URL will be `http://localhost:8000/` (as the server was started on port `8000`). Jupyter will automatically replace this URL with the correct one during requests.

In the case where the player is opened in a separate tab, we need to obtain an external URL for the Colab virtual machine. To accomplish this, we use Colab's helper function [eval_js](https://github.com/googlecolab/colabtools/blob/0e3c20fb16bf1891d62b4db67645902a843e186a/google/colab/output/_js.py#L23) to interact with JavaScript within the current cell's context by executing Colab’s JS function [proxyPort](https://github.com/googlecolab/colabtools/blob/a601b1fdde246573c78fc002b931e0b1ea96fcd7/packages/outputframe/lib/index.d.ts#L30). It will return the current server URL. Be aware that your browser may need to allow third-party cookies for this to function properly.

```python
# Get get an external URL to the virtual machine
from google.colab.output import eval_js
url = eval_js("google.colab.kernel.proxyPort(8000)")
```

Furthermore, it includes a workaround fixing [A UTF-8 locale is required. Got ANSI_X3.4-1968](https://github.com/pyannote/pyannote-audio/issues/1269) problem, which occurs after installing Pyannote. For unknown reasons, the `locale` is being set to `ANSI_X3.4-1968` and as a temporary solution, we can override [locale.getpreferredencoding](https://docs.python.org/3/library/locale.html#locale.getpreferredencoding) function as follows:

```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```

Lastly, the `process` function is responsible for preparing the video file, as well as the video player templates, and invoking functions that are defined in the Main code section to generate subtitles.

## Conclusion

To sum up, this tutorial provides a comprehensive guide to building a speech recognition and speaker diarization pipeline utilizing three different models. 

With the recent release of the [Whisper API](https://openai.com/blog/introducing-chatgpt-and-whisper-apis) developers can now easily integrate the model into their apps.

Additionally, Whisper has recently been [ported to C++](https://github.com/ggerganov/whisper.cpp), which enables high-performance inferencing. This makes it feasible to execute the model on a CPU or even on a Raspberry Pi. Moreover, thanks to WebAssembly (more details on this in my post on [Python on a Webpage](https://karay.me/2022/07/12/bringing-python-to-the-web.html)), the model can also run within a web browser.  You can try out Whisper in the browser for yourself in this [live demo](https://whisper.ggerganov.com/).