import unittest
from voice_tools import VoiceTrainer
import os
import numpy as np
import soundfile as sf

class TestVoiceTrainer(unittest.TestCase):

    def setUp(self):
        """Set up a temporary dataset directory for testing."""
        self.character_name = "TestCharacter"
        self.trainer = VoiceTrainer(character_name=self.character_name)
        os.makedirs(self.trainer.dataset_path, exist_ok=True)
        os.makedirs(self.trainer.output_path, exist_ok=True)

        # Create a valid test audio file
        self.test_audio_path = os.path.join(self.trainer.dataset_path, "test.wav")
        sample_rate = 22050
        duration = 1  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # Generate a sine wave
        sf.write(self.test_audio_path, audio_data, sample_rate)

    def tearDown(self):
        """Clean up the temporary dataset directory."""
        for root, dirs, files in os.walk(self.trainer.dataset_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.trainer.dataset_path)

        for root, dirs, files in os.walk(self.trainer.output_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.trainer.output_path)

    def test_augment_audio(self):
        """Test the audio augmentation method."""
        self.trainer.augment_audio(self.test_audio_path)
        augmented_file = self.test_audio_path.replace(".wav", "_augmented.wav")
        self.assertTrue(os.path.exists(augmented_file))

    def test_trim_silence(self):
        """Test the silence trimming method."""
        self.trainer.trim_silence(self.test_audio_path)
        trimmed_file = self.test_audio_path.replace(".wav", "_trimmed.wav")
        self.assertTrue(os.path.exists(trimmed_file))

    def test_validate_metadata(self):
        """Test the metadata validation method."""
        metadata_file = os.path.join(self.trainer.dataset_path, "metadata.csv")
        with open(metadata_file, "w") as f:
            f.write("missing_file.wav|Some text\n")

        self.trainer.validate_metadata()
        with open(metadata_file, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 0)

    def test_check_audio_quality(self):
        """Test the audio quality check method."""
        # This is a placeholder test; actual implementation would require real audio data
        self.trainer.check_audio_quality(self.test_audio_path)

    def test_dataset_statistics(self):
        """Test the dataset statistics method."""
        metadata_file = os.path.join(self.trainer.dataset_path, "metadata.csv")
        with open(metadata_file, "w") as f:
            f.write("test.wav|Some text\n")

        self.trainer.dataset_statistics()

if __name__ == "__main__":
    unittest.main()