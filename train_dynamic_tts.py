import os
import argparse
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager

def train_tts_model(model_name, dataset_path, output_path, use_phonemes=True, phoneme_language="en-us", multi_speaker=False):
    # Define dataset configuration
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech" if not multi_speaker else "vctk",
        meta_file_train=os.path.join(dataset_path, "metadata.csv"),
        meta_file_val=os.path.join(dataset_path, "valid.csv"),
        path=os.path.join(dataset_path, "wavs")
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
        use_phonemes=use_phonemes,
        phoneme_language=phoneme_language,
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        use_speaker_embedding=multi_speaker
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

    # Initialize speaker manager for multi-speaker training
    speaker_manager = None
    if multi_speaker:
        speaker_manager = SpeakerManager()
        speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
        config.num_speakers = speaker_manager.num_speakers

    # Initialize the model
    model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

    # Initialize the trainer
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # Start training
    trainer.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TTS model dynamically.")
    parser.add_argument("--model", type=str, default="GlowTTS", help="Model name (e.g., GlowTTS)")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save outputs")
    parser.add_argument("--use-phonemes", action="store_true", help="Enable phoneme-based training")
    parser.add_argument("--phoneme-language", type=str, default="en-us", help="Phoneme language")
    parser.add_argument("--multi-speaker", action="store_true", help="Enable multi-speaker training")
    args = parser.parse_args()

    train_tts_model(
        model_name=args.model,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        use_phonemes=args.use_phonemes,
        phoneme_language=args.phoneme_language,
        multi_speaker=args.multi_speaker
    )