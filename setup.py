from setuptools import setup

setup(
    name="speech_to_music",
    packages=["speech_to_music"],
    install_requires=[
        'youtube-dl==2021.6.6',
        'transformers==4.12.5',
        'huggingface-hub==0.2.1',
        'numpy==1.17.3',
        'librosa >= 0.8',
        'pytorch-lightning==1.5.10', # important this version!
        'torchaudio_augmentations==0.2.1', # for augmentation
        'audiomentations==0.22.0',
        'einops',
        'langdetect==1.0.8',
        'sklearn',
        'wandb==0.12.0',
        'gensim==3.8.3',
        'umap-learn==0.5.2',
        'gradio',
        'pandas',
        'omegaconf'
    ]
)