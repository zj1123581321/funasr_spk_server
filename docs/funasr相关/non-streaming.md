### Speech Recognition (Non-streaming)

[](https://github.com/modelscope/FunASR#speech-recognition-non-streaming)

#### SenseVoice

[](https://github.com/modelscope/FunASR#sensevoice)

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
```

Parameter Description:

- `model_dir`: The name of the model, or the path to the model on the local disk.
- `vad_model`: This indicates the activation of VAD (Voice Activity Detection). The purpose of VAD is to split long audio into shorter clips. In this case, the inference time includes both VAD and SenseVoice total consumption, and represents the end-to-end latency. If you wish to test the SenseVoice model's inference time separately, the VAD model can be disabled.
- `vad_kwargs`: Specifies the configurations for the VAD model. `max_single_segment_time`: denotes the maximum duration for audio segmentation by the `vad_model`, with the unit being milliseconds (ms).
- `use_itn`: Whether the output result includes punctuation and inverse text normalization.
- `batch_size_s`: Indicates the use of dynamic batching, where the total duration of audio in the batch is measured in seconds (s).
- `merge_vad`: Whether to merge short audio fragments segmented by the VAD model, with the merged length being `merge_length_s`, in seconds (s).
- `ban_emo_unk`: Whether to ban the output of the `emo_unk` token.