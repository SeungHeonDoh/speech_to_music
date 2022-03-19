# Speech to Music through Emotion

This project maps `speech` and `music` to the same embedding space and supports music item search for speech query by calculating the similarity between them.


## Reference
Speech to Music through Emotion, Interspeech 2022 [[will be updated]()]
-- SeungHeon Doh, Minz Won, Keunwoo Choi, Juhan Nam


## Quickstart
Requirements: >1 GPU

```
pip3 install -e .
cd scripts
bash download_from_hdfs.sh
cd ../speech_to_music/metirc_learning
python3 train.py
```

You can use `wandb` to view training logs, which are stored in `./exp` by default.


## Data
- You need to collect audio files of AudioSet mood subset [[link](https://research.google.com/audioset/ontology/music_mood_1.html)].
- Read the audio files and store them into `.npy` format.
- Other relevant data including IEMOCAP ([original link](https://sail.usc.edu/iemocap/))
- Pretrained models will be updated [[link]()].


## Results
Based on the Audioset and IEMOCAP datasets, the following results are obtained.
This project can represent speech of various modalities with audio, text, and fusion.

## Model Summary
- Word : glove embedding
- Music : Tagging Transformer
- Speech : Wav2vec 2.0

Each backbone model is extracted with 64-dimensional embedding through the projection model.


## Embedding Inference
```
cd metric_learning
infer.py --inference_type music_extractor --music_path {your_music_path}
infer.py --inference_type speech_extractor --speech_path {your_speech_path}
```

The output is represented as a 64-dimensional semantic vector on joint embeddings.