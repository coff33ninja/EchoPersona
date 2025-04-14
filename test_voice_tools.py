import unittest
from voice_tools import VoiceTrainer
import os

class TestVoiceTrainer(unittest.TestCase):

    def setUp(self):
        """Set up a temporary dataset directory for testing."""
        self.dataset_path = "test_dataset"
        self.trainer = VoiceTrainer(dataset_path=self.dataset_path)
        os.makedirs(self.dataset_path, exist_ok=True)

    def tearDown(self):
        """Clean up the temporary dataset directory."""
        for root, dirs, files in os.walk(self.dataset_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.dataset_path)

    def test_augment_audio(self):
        """Test the audio augmentation method."""
        test_file = os.path.join(self.dataset_path, "test.wav")
        with open(test_file, "wb") as f:
            f.write(b"dummy audio data")

        self.trainer.augment_audio(test_file)
        augmented_file = test_file.replace(".wav", "_augmented.wav")
        self.assertTrue(os.path.exists(augmented_file))

    def test_trim_silence(self):
        """Test the silence trimming method."""
        test_file = os.path.join(self.dataset_path, "test.wav")
        with open(test_file, "wb") as f:
            f.write(b"dummy audio data")

        self.trainer.trim_silence(test_file)
        trimmed_file = test_file.replace(".wav", "_trimmed.wav")
        self.assertTrue(os.path.exists(trimmed_file))

    def test_validate_metadata(self):
        """Test the metadata validation method."""
        metadata_file = os.path.join(self.dataset_path, "metadata.csv")
        with open(metadata_file, "w") as f:
            f.write("missing_file.wav|Some text\n")

        self.trainer.validate_metadata()
        with open(metadata_file, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 0)

    def test_check_audio_quality(self):
        """Test the audio quality check method."""
        test_file = os.path.join(self.dataset_path, "test.wav")
        with open(test_file, "wb") as f:
            f.write(b"dummy audio data")

        # This is a placeholder test; actual implementation would require real audio data
        self.trainer.check_audio_quality(test_file)

    def test_dataset_statistics(self):
        """Test the dataset statistics method."""
        metadata_file = os.path.join(self.dataset_path, "metadata.csv")
        with open(metadata_file, "w") as f:
            f.write("test.wav|Some text\n")

        test_file = os.path.join(self.dataset_path, "test.wav")
        with open(test_file, "wb") as f:
            f.write(b"dummy audio data")

        self.trainer.dataset_statistics()

if __name__ == "__main__":
    unittest.main()