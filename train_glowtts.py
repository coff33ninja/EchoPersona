import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Define output path
output_path = os.path.dirname(os.path.abspath(__file__))

# Define dataset configuration
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="voice_datasets/Zhongli/metadata.csv",
    meta_file_val="voice_datasets/Zhongli/valid.csv",
    path="voice_datasets/Zhongli/wavs"
)

# Initialize training configuration
config = GlowTTSConfig(
    batch_size=16,
    eval_batch_size=8,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
)

# Initialize the audio processor
ap = AudioProcessor.init_from_config(config)

# Initialize the tokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)

# Load data samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Initialize the model
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# Initialize the trainer
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# Start training
trainer.fit()