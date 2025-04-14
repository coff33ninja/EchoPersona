import speech_recognition as sr
import pyttsx3
import sounddevice as sd
import tempfile
import os
import numpy as np
from typing import Union, Optional, List, Tuple
import logging
import time
from io import BytesIO
from scipy.io.wavfile import write as write_wav, read as read_wav
from gtts import gTTS
import torch
from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import speedup
import simpleaudio as sa
import shutil
import random
import librosa
import noisereduce as nr
import soundfile as sf # Used for saving processed numpy array to WAV
import io             # To handle audio data in memory
import csv            # For reading/writing metadata
import re             # <<< FIX: Import the regular expression module >>>

# --- TTS Library Import ---
# Try importing TTS, handle potential ImportError
try:
    from TTS.api import TTS as CoquiTTS
    from TTS.utils.audio import AudioProcessor # Keep if used elsewhere
except ImportError:
    print("=" * 50)
    print("WARNING: TTS library not found.")
    print("Please install it using: pip install TTS")
    print("Voice cloning and Coqui TTS features will be unavailable.")
    print("=" * 50)
    CoquiTTS = None  # Set to None if import fails
    AudioProcessor = None # Set to None if import fails

# --- Constants ---
BASE_DATASET_DIR = "voice_datasets" # Base directory for all character datasets
BASE_MODEL_DIR = "trained_models"   # Base directory for all trained character models
METADATA_FILENAME = "metadata.csv"  # Standard metadata filename

# --- Helper Functions ---

# Function to play audio using pydub
def play_audio_with_pydub(file_path):
    """
    Plays an audio file using pydub. Handles various formats.

    Args:
        file_path: Path to the audio file to play.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        print(f"Attempting to play audio with pydub: {file_path}")
        play(audio)
        print(f"Audio playback finished for: {file_path}")
    except Exception as e:
        print(f"Error playing audio with pydub: {e}")
        raise # Re-raise exception to trigger fallback

# Function to play audio using simpleaudio (expects WAV)
def play_audio_with_simpleaudio(file_path):
    """
    Plays a WAV audio file using simpleaudio.

    Args:
        file_path: Path to the WAV audio file to play.
    """
    try:
        print(f"Attempting to play audio with simpleaudio: {file_path}")
        if not file_path.lower().endswith(".wav"):
             print("Simpleaudio fallback requires WAV format. Skipping playback.")
             return
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait until playback is finished
        print(f"Audio playback finished for: {file_path}")
    except Exception as e:
        print(f"Error playing audio with simpleaudio: {e}")

# Updated logic to use pydub as primary and simpleaudio as fallback (for WAV)
def play_audio(file_path):
    """
    Plays an audio file using pydub or simpleaudio as fallback (for WAV).

    Args:
        file_path: Path to the audio file to play.
    """
    try:
        play_audio_with_pydub(file_path)
    except Exception as e:
        print(f"Pydub playback failed ({e}). Falling back to simpleaudio for WAV if applicable.")
        try:
            play_audio_with_simpleaudio(file_path)
        except Exception as fallback_e:
            print(f"Simpleaudio fallback also failed: {fallback_e}")
            print("Ensure you have playback libraries and potentially ffmpeg installed.")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Utility function to demonstrate the use of `time`
def measure_execution_time(func):
    """ Decorator to measure the execution time of a function. """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function '{func.__name__}' executed in: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Function to use `scipy.io.wavfile.read` for reading WAV files
def read_wav_file_scipy(file_path):
    """ Reads a WAV file using Scipy and returns the sample rate and data. """
    try:
        sample_rate, data = read_wav(file_path)
        logging.debug(f"Read WAV file (SciPy): {file_path} (Sample Rate: {sample_rate}, Data Shape: {data.shape})")
        return sample_rate, data
    except Exception as e:
        logging.error(f"Error reading WAV file with SciPy: {e}")
        return None, None

# Example usage of AudioProcessor (if TTS is installed)
@measure_execution_time
def process_audio_sample_coqui(audio_file):
    """ Processes an audio file using Coqui TTS AudioProcessor. """
    if AudioProcessor is None:
        print("AudioProcessor not available (TTS library missing).")
        return None
    try:
        # Example config - adjust as needed for your TTS model
        ap = AudioProcessor(sample_rate=22050, num_mels=80)
        processed_audio = ap.load_wav(audio_file)
        logging.info("Audio processing with Coqui AudioProcessor complete.")
        return processed_audio
    except Exception as e:
        logging.error(f"Error processing audio with Coqui AudioProcessor: {e}")
        return None

# --- SpeechToText Class (Largely Unchanged - STT is generally character-agnostic) ---
class SpeechToText:
    """
    Handles Speech-to-Text (STT) with audio preprocessing and engine selection.
    Includes noise reduction, normalization, and resampling for improved accuracy.
    Supports engines: 'google', 'sphinx', 'whisper', 'vosk'.
    """

    def __init__(
        self,
        use_microphone: bool = True,
        audio_file: Optional[str] = None,
        sample_rate: int = 44100, # Original sample rate for recording if using mic
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
        hotword: Optional[str] = None,
        timeout: Optional[int] = None,
        target_sr: int = 16000, # Target sample rate for STT engine
        engine: str = "google", # STT engine: 'google', 'sphinx', 'whisper', 'vosk'
        vosk_model_path: Optional[str] = None, # Required if engine='vosk'
        whisper_model_size: str = "base" # Whisper model size: tiny, base, small, medium, large
    ) -> None:
        """
        Initializes the SpeechToText object.

        Args:
            use_microphone: Use the microphone as the audio source.
            audio_file: Path to an audio file to transcribe.
            sample_rate: Sample rate for microphone recording.
            chunk_size: Size of audio chunks for processing.
            device_index: Index of the input device (None for default).
            hotword: Optional hotword to listen for.
            timeout: Optional timeout for listen() operation (seconds).
            target_sr: Sample rate to convert audio to before STT (e.g., 16000).
            engine: STT engine ('google', 'sphinx', 'whisper', 'vosk'). Default: 'google'.
            vosk_model_path: Path to the downloaded Vosk model directory (required for 'vosk').
            whisper_model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large'). Default: 'base'.
        """
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.mic_sample_rate = sample_rate
        self.target_sr = target_sr
        self.engine = engine.lower()
        self.vosk_model_path = vosk_model_path
        self.whisper_model_size = whisper_model_size

        # Validate engine choice
        allowed_engines = ["google", "sphinx", "whisper", "vosk"]
        if self.engine not in allowed_engines:
            raise ValueError(f"Invalid engine '{self.engine}'. Choose from: {allowed_engines}")
        if self.engine == "vosk" and not self.vosk_model_path:
            logging.warning("Engine set to 'vosk' but 'vosk_model_path' not provided. Transcription will likely fail.")
            # Consider raising ValueError instead if path is strictly required
        elif self.engine == "whisper":
             # Check if whisper is installed
             try:
                 from importlib.util import find_spec
                 if not find_spec("whisper"):
                     raise ImportError("Whisper engine requires 'openai-whisper' library. Please install it: pip install -U openai-whisper")
             except ImportError:
                 logging.error("Whisper engine selected, but 'openai-whisper' library not found. Please install it: pip install -U openai-whisper")
                 # Optionally raise error or default to another engine
                 # raise ImportError("Whisper engine requires 'openai-whisper' library.")

        if use_microphone:
            try:
                self.microphone = sr.Microphone(
                    sample_rate=self.mic_sample_rate,
                    chunk_size=chunk_size,
                    device_index=device_index,
                )
                # Adjust for ambient noise immediately after initializing the mic
                self.adjust_for_ambient_noise()
            except Exception as e:
                logging.error(f"Failed to initialize microphone (device index: {device_index}): {e}")
        self.use_microphone = use_microphone and (self.microphone is not None)
        self.audio_file = audio_file
        self.chunk_size = chunk_size
        self.hotword = hotword
        self.timeout = timeout


    def adjust_for_ambient_noise(self, duration: float = 1.0) -> None:
        """ Adjusts the recognizer for ambient noise levels. """
        if self.use_microphone and self.microphone:
            logging.info("Adjusting for ambient noise...")
            try:
                with self.microphone as source:
                    self.recognizer.dynamic_energy_threshold = True # Enable dynamic adjustment
                    self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                logging.info(f"Ambient noise adjustment complete. Energy threshold: {self.recognizer.energy_threshold:.2f}")
            except Exception as e:
                logging.error(f"Could not complete ambient noise adjustment: {e}")
        elif self.use_microphone:
            logging.warning("Microphone specified but not available for ambient noise adjustment.")


    def record_audio(
        self, duration: Optional[float] = None, listen_for_silence: bool = False
    ) -> Union[sr.AudioData, None]:
        """
        Records audio from the microphone or reads from a file into an AudioData object.
        """
        if self.use_microphone and self.microphone:
            # Note: adjust_for_ambient_noise is called in __init__ now
            with self.microphone as source:
                if self.hotword:
                    logging.info(f"Listening for hotword: '{self.hotword}'")
                    try:
                        # Listen for a short phrase to detect the hotword
                        audio_chunk = self.recognizer.listen(
                            source, phrase_time_limit=5, timeout=self.timeout
                        )
                        # Use a quick, less resource-intensive check if possible (e.g., Sphinx if configured)
                        # Using Google here for simplicity, but be mindful of API calls
                        detected_text = self.recognizer.recognize_google(audio_chunk, show_all=False)
                        if self.hotword.lower() not in detected_text.lower():
                            logging.info("Hotword not detected.")
                            return None
                        logging.info("Hotword detected!")
                    except sr.WaitTimeoutError:
                        logging.warning("Timeout while listening for hotword.")
                        return None
                    except sr.UnknownValueError:
                        logging.info("Could not understand audio while listening for hotword.")
                        return None # Assume hotword not detected if unintelligible
                    except sr.RequestError as e:
                        logging.error(f"API Error during hotword check: {e}")
                        return None
                    except Exception as e:
                        logging.error(f"Error listening for hotword: {e}")
                        return None

                logging.info("Recording main audio...")
                audio = None
                try:
                    if duration is not None:
                        audio = self.recognizer.listen(
                            source, timeout=self.timeout, phrase_time_limit=duration
                        )
                        logging.info(f"Finished recording ({duration} seconds).")
                    elif listen_for_silence:
                        # listen() naturally stops on silence based on energy levels
                        audio = self.recognizer.listen(source, timeout=self.timeout)
                        logging.info("Finished recording (silence detected or timeout).")
                    else:
                        # Listen indefinitely until timeout (if set) or manual stop
                        audio = self.recognizer.listen(source, timeout=self.timeout)
                        logging.info("Finished recording (indefinite/timeout).")
                    return audio
                except sr.WaitTimeoutError:
                    logging.warning("Timeout during main audio recording.")
                    return None
                except Exception as e:
                    logging.error(f"Error during main audio recording: {e}")
                    return None

        elif self.audio_file:
            if not os.path.exists(self.audio_file):
                 logging.error(f"Audio file not found: {self.audio_file}")
                 return None
            logging.info(f"Reading audio from file: {self.audio_file}")
            try:
                with sr.AudioFile(self.audio_file) as source:
                    audio = self.recognizer.record(source) # Record entire file
                logging.info("Finished reading audio file.")
                return audio
            except FileNotFoundError: # Should be caught above, but double check
                logging.error(f"Audio file not found: {self.audio_file}")
                return None
            except Exception as e:
                logging.error(f"Error reading audio file '{self.audio_file}': {e}")
                return None
        else:
            logging.error("No audio source specified or microphone initialization failed.")
            return None

    @measure_execution_time # Measure preprocessing time
    def _preprocess_audio(self, audio_data: sr.AudioData) -> Optional[Tuple[np.ndarray, int]]:
        """ Internal method: preprocess AudioData (load, resample, denoise, normalize). """
        try:
            logging.info("Starting audio preprocessing...")
            # 1. Get raw WAV data bytes and original sample rate
            wav_bytes = audio_data.get_wav_data(convert_rate=None, convert_width=None)
            original_sr = audio_data.sample_rate
            logging.debug(f"Original sample rate: {original_sr} Hz")

            # 2. Load WAV bytes into NumPy array using librosa
            # Use BytesIO to load from memory
            audio_float, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True) # Load original SR, ensure mono
            logging.debug(f"Loaded audio shape: {audio_float.shape}")

            # 3. Resample if necessary
            if sr != self.target_sr:
                logging.info(f"Resampling audio from {sr} Hz to {self.target_sr} Hz...")
                audio_float = librosa.resample(audio_float, orig_sr=sr, target_sr=self.target_sr)
                current_sr = self.target_sr
                logging.debug(f"Resampled audio shape: {audio_float.shape}")
            else:
                current_sr = sr
                logging.info("Audio already at target sample rate.")

            # 4. Apply Noise Reduction (Optional - can be intensive)
            logging.info("Applying noise reduction (spectral gating)...")
            # Using non-stationary is generally better if noise varies
            reduced_noise_audio = nr.reduce_noise(y=audio_float, sr=current_sr, stationary=False)
            logging.debug("Noise reduction complete.")
            # Use original audio if skipping noise reduction
            reduced_noise_audio = audio_float

            # 5. Normalize Audio to range [-1, 1]
            logging.info("Normalizing audio...")
            normalized_audio = librosa.util.normalize(reduced_noise_audio)
            logging.debug("Normalization complete.")

            logging.info("Audio preprocessing finished successfully.")
            return normalized_audio, current_sr

        except Exception as e:
            logging.error(f"Error during audio preprocessing: {e}", exc_info=True)
            return None

    @measure_execution_time # Measure transcription time
    def transcribe_audio(
        self, audio: sr.AudioData, language: str = "en-US"
    ) -> Optional[str]:
        """
        Preprocesses and transcribes audio data using the selected engine.

        Args:
            audio: The raw AudioData object to preprocess and transcribe.
            language: Language code for transcription (e.g., "en-US", "es-ES").
                      Format might vary slightly depending on the engine.

        Returns:
            The transcribed text, or None on error.
        """
        if not audio:
            logging.error("No audio data provided for transcription.")
            return None

        # --- Preprocess the Audio ---
        # Preprocessing might not always be needed or desired, especially for high-quality inputs
        # Consider making preprocessing optional based on a flag
        # For now, we assume preprocessing is beneficial
        # processed_data = self._preprocess_audio(audio)
        # if processed_data is None:
        #     logging.error("Audio preprocessing failed. Cannot transcribe.")
        #     return None
        # processed_audio_np, processed_sr = processed_data

        # --- Transcribe the ORIGINAL audio (or preprocessed if enabled) ---
        # Using original audio directly with speech_recognition for simplicity now
        # If using preprocessed numpy array, need to save to temp file first as before

        text = None
        try:
            # --- Choose Recognition Engine based on self.engine ---
            logging.info(f"Attempting transcription using engine: '{self.engine}' (Language: {language})")

            if self.engine == "google":
                # Uses the raw AudioData directly
                text = self.recognizer.recognize_google(audio, language=language)

            elif self.engine == "whisper":
                try:
                    # Requires: pip install -U openai-whisper
                    # Models are downloaded automatically on first use to ~/.cache/whisper
                    whisper_lang = language.split('-')[0] if '-' in language else language # Whisper uses 'en', 'es', etc.
                    logging.info(f"Using Whisper model size: {self.whisper_model_size}")
                    # Uses the raw AudioData directly
                    text = self.recognizer.recognize_whisper(
                        audio,
                        model=self.whisper_model_size,
                        language=whisper_lang
                    )
                except ImportError:
                    logging.error("Whisper not installed. Run 'pip install -U openai-whisper'")
                except Exception as whisper_e:
                    logging.error(f"Error during Whisper transcription: {whisper_e}", exc_info=True)

            elif self.engine == "vosk":
                try:
                    # Requires: pip install vosk
                    # Requires manual download of Vosk model for the target language.
                    # Download from: https://alphacephei.com/vosk/models
                    if not self.vosk_model_path or not os.path.exists(self.vosk_model_path):
                         logging.error(f"Vosk model path '{self.vosk_model_path}' not provided or not found. Cannot use Vosk.")
                    else:
                         logging.info(f"Using Vosk model from: {self.vosk_model_path}")
                         # Note: speech_recognition's vosk wrapper might expect language codes like 'en-us'
                         # Uses the raw AudioData directly
                         text = self.recognizer.recognize_vosk(
                             audio,
                             language=language, # Pass full language code
                             # The library should find the model if vosk is configured correctly,
                             # but explicitly passing path might be needed if default doesn't work.
                             # Check speech_recognition docs for exact vosk integration details.
                             # model_path=self.vosk_model_path # May need to uncomment/adjust
                         )
                except ImportError:
                    logging.error("Vosk not installed. Run 'pip install vosk'")
                except Exception as vosk_e:
                    logging.error(f"Error during Vosk transcription: {vosk_e}", exc_info=True)

            elif self.engine == "sphinx":
                try:
                    # Requires: pocketsphinx, potentially language packs
                    # Uses the raw AudioData directly
                    text = self.recognizer.recognize_sphinx(audio, language=language)
                except ImportError:
                    logging.error("CMU Sphinx (pocketsphinx) not installed/configured.")
                except Exception as sphinx_e:
                    logging.error(f"Error during Sphinx transcription: {sphinx_e}", exc_info=True)

            # --- Log results ---
            if text:
                 logging.info("Transcription successful.")
                 logging.info(f"Recognized Text: {text}")
            else:
                 # Specific errors logged above, this catches cases where no text was returned
                 logging.warning(f"Transcription using '{self.engine}' failed or returned empty text.")
            return text

        except sr.UnknownValueError:
            logging.warning(f"Speech Recognition ({self.engine}) could not understand the audio.")
            return None
        except sr.RequestError as e:
            # Specific to online APIs like Google
            logging.error(f"API request failed for engine '{self.engine}': {e}")
            return None
        except Exception as e:
            # Catch-all for other unexpected errors during transcription phase
            logging.error(f"Unexpected error during transcription with '{self.engine}': {e}", exc_info=True)
            return None
        # No finally block needed if not using temp files


    def process_audio(
        self,
        duration: Optional[float] = None,
        listen_for_silence: bool = False,
        language: str = "en-US",
    ) -> Optional[str]:
        """
        Records (if mic) or loads (if file), then transcribes audio using the configured engine.
        Preprocessing is currently bypassed in transcribe_audio for simplicity.

        Args:
            duration: The duration to record (seconds). None for silence detection/timeout.
            listen_for_silence: Stop recording on silence (if duration is None).
            language: Language code (e.g., "en-US").

        Returns:
            The transcribed text, or None on error.
        """
        # Record or load audio
        audio = self.record_audio(duration, listen_for_silence)
        if audio is None:
            # record_audio logs specific errors (file not found, mic error, etc.)
            logging.warning("Recording/loading failed or produced no audio data.")
            return None

        # Transcribe using the configured engine
        return self.transcribe_audio(audio, language)


# --- TextToSpeech Class (Standard TTS - Largely Unchanged) ---
class TextToSpeech:
    """ Handles standard Text-to-Speech (TTS) using pyttsx3 or gTTS. """
    def __init__(
        self,
        use_pyttsx3: bool = True,
        voice_name: Optional[str] = None,
        speech_rate: int = 150,
        lang: str = "en",
    ) -> None:
        self.use_pyttsx3 = use_pyttsx3
        self.voice_name = voice_name
        self.speech_rate = speech_rate
        self.lang = lang
        self.engine = None

        if self.use_pyttsx3:
            try:
                self.engine = pyttsx3.init()
                if self.voice_name:
                    available_voices = self.engine.getProperty("voices")
                    selected_voice_id = None
                    for voice in available_voices:
                        if self.voice_name.lower() in voice.name.lower():
                            selected_voice_id = voice.id
                            logging.info(f"Found matching pyttsx3 voice: {voice.name}")
                            break
                    if selected_voice_id:
                        self.engine.setProperty("voice", selected_voice_id)
                    else:
                        logging.warning(f"pyttsx3 voice containing '{self.voice_name}' not found. Using default.")
                self.engine.setProperty("rate", self.speech_rate)
            except Exception as e:
                logging.error(f"Error initializing pyttsx3 engine: {e}")
                self.engine = None
                self.use_pyttsx3 = False
                logging.warning("Disabling pyttsx3 due to initialization error.")

    def speak(
        self, text: str, play_audio_flag: bool = True, filename: Optional[str] = None
    ) -> Union[None, bytes]:
        """ Converts text to speech -> plays, saves, or returns bytes. """
        if self.use_pyttsx3 and self.engine:
            try:
                if filename:
                    logging.info(f"Saving speech (pyttsx3) to: {filename}")
                    self.engine.save_to_file(text, filename)
                    self.engine.runAndWait()
                    return None
                elif play_audio_flag:
                    logging.info(f"Speaking (pyttsx3): {text[:50]}...") # Log snippet
                    self.engine.say(text)
                    self.engine.runAndWait()
                    return None
                else:
                    # pyttsx3 doesn't directly return bytes. Save to temp file.
                    logging.info("Generating speech data (pyttsx3)...")
                    temp_filename = None
                    try:
                        # Use BytesIO to avoid disk write if possible, though pyttsx3 lacks direct support
                        # Fallback to temp file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            temp_filename = tmp_file.name
                        self.engine.save_to_file(text, temp_filename)
                        self.engine.runAndWait()
                        with open(temp_filename, "rb") as f:
                            audio_data = f.read()
                        return audio_data
                    finally:
                        if temp_filename and os.path.exists(temp_filename):
                            try:
                                os.remove(temp_filename)
                            except OSError as e:
                                logging.error(f"Error removing temp pyttsx3 file {temp_filename}: {e}")
            except Exception as e:
                logging.error(f"Error during pyttsx3 operation: {e}")
                return None

        elif not self.use_pyttsx3: # use gTTS
            try:
                logging.info(f"Using gTTS (lang={self.lang}) for: {text[:50]}...")
                tts = gTTS(text=text, lang=self.lang)
                audio_data = BytesIO()
                tts.write_to_fp(audio_data)
                audio_data.seek(0)
                audio_bytes = audio_data.read()

                if filename:
                    logging.info(f"Saving speech (gTTS) to: {filename}")
                    with open(filename, "wb") as f:
                        f.write(audio_bytes)
                    return None
                elif play_audio_flag:
                    logging.info("Playing gTTS audio...")
                    # Need to play MP3 bytes. Use pydub/simpleaudio via helper.
                    temp_mp3_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                             temp_mp3_path = tmp_file.name
                             tmp_file.write(audio_bytes)

                        play_audio(temp_mp3_path) # Use helper function

                    except Exception as e:
                        logging.error(f"Error during gTTS playback attempt: {e}")
                    finally:
                         if temp_mp3_path and os.path.exists(temp_mp3_path):
                              try:
                                  os.remove(temp_mp3_path)
                              except OSError as e:
                                  logging.error(f"Error removing temp gTTS file {temp_mp3_path}: {e}")
                    return None
                else:
                    logging.info("Returning speech data (gTTS bytes)...")
                    return audio_bytes
            except Exception as e:
                logging.error(f"Error creating or processing gTTS object: {e}")
                return None
        else:
            logging.error("TTS engine not available.")
            return None


# --- ClonedVoiceTTS Class (For pre-trained models like XTTS - Largely Unchanged) ---
class ClonedVoiceTTS:
    """ Handles Text-to-Speech (TTS) using voice cloning models (e.g., XTTS). """
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        reference_wavs: Union[str, List[str]] = None,
        use_gpu: bool = True,
    ):
        if CoquiTTS is None:
            raise ImportError("TTS library is required for ClonedVoiceTTS but not found.")

        self.model_name = model_name
        self.reference_wavs = reference_wavs
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.tts_engine = None

        if not self.reference_wavs:
            raise ValueError("Reference WAV file(s) must be provided for voice cloning.")
        if isinstance(self.reference_wavs, str):
            self.reference_wavs = [self.reference_wavs]
        for wav_path in self.reference_wavs:
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"Reference WAV file not found: {wav_path}")

        try:
            logging.info(f"Loading TTS voice cloning model: {self.model_name} (GPU: {self.use_gpu})")
            # Ensure CoquiTTS is initialized correctly
            self.tts_engine = CoquiTTS(model_name=self.model_name, gpu=self.use_gpu)
            logging.info("TTS voice cloning model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load TTS model '{self.model_name}': {e}", exc_info=True)
            self.tts_engine = None

    def speak(
        self,
        text: str,
        language: str = "en",
        output_filename: str = "cloned_output.wav",
        play_audio_flag: bool = True,
    ) -> None:
        """ Generates speech using the cloned voice, saves it, optionally plays it. """
        if not self.tts_engine:
            logging.error("TTS voice cloning engine not loaded. Cannot synthesize.")
            return

        try:
            logging.info(f"Generating cloned speech for: '{text[:50]}...' (Lang: {language})")
            # Use tts_to_file with speaker_wav argument
            self.tts_engine.tts_to_file(
                text=text,
                speaker_wav=self.reference_wavs,
                language=language,
                file_path=output_filename,
            )
            logging.info(f"Cloned speech saved to: {output_filename}")

            if play_audio_flag:
                logging.info("Playing cloned audio...")
                try:
                    play_audio(output_filename) # Use helper function
                except Exception as e:
                    logging.error(f"Error playing cloned audio file '{output_filename}': {e}")

        except Exception as e:
            logging.error(f"Error during cloned TTS synthesis: {e}", exc_info=True)


# --- Voice Trainer Class (Character Aware) ---
class VoiceTrainer:
    """ Handles voice training dataset management and interaction with training scripts for specific characters. """
    def __init__(self, character_name: str, base_dataset_dir: str = BASE_DATASET_DIR, base_model_dir: str = BASE_MODEL_DIR):
        """
        Initializes the VoiceTrainer for a specific character.

        Args:
            character_name: The name of the character (used for directory paths).
            base_dataset_dir: The root directory containing all character datasets.
            base_model_dir: The root directory containing all trained character models.
        """
        if not character_name:
            raise ValueError("Character name cannot be empty.")
        # Sanitize character name for use in paths
        # <<< FIX: Uses 're' module which needs to be imported >>>
        self.character_name = re.sub(r'[\\/*?:"<>|]', "_", character_name)
        self.base_dataset_dir = base_dataset_dir
        self.base_model_dir = base_model_dir

        # Define character-specific paths
        self.dataset_path = os.path.join(self.base_dataset_dir, self.character_name)
        self.output_path = os.path.join(self.base_model_dir, self.character_name)
        self.metadata_path = os.path.join(self.dataset_path, METADATA_FILENAME)

        logging.info(f"Initializing VoiceTrainer for character: '{self.character_name}'")
        logging.info(f"  Dataset Path: {self.dataset_path}")
        logging.info(f"  Model Output Path: {self.output_path}")
        logging.info(f"  Metadata Path: {self.metadata_path}")

        # Ensure directories exist
        try:
            os.makedirs(self.dataset_path, exist_ok=True)
            os.makedirs(self.output_path, exist_ok=True)
        except OSError as e:
             logging.error(f"Error creating directories for character '{self.character_name}': {e}")
             # Depending on severity, might want to raise the error

        # Create metadata file with header if it doesn't exist in the character's dataset path
        self._ensure_metadata_header()

    def _ensure_metadata_header(self):
        """Ensures the metadata file exists and has the correct header."""
        header = "audio_file|text|normalized_text\n"  # LJSpeech format: wav_filename|transcription|normalized_transcription
        try:
            if not os.path.exists(self.metadata_path):
                 with open(self.metadata_path, "w", encoding="utf-8") as f:
                      f.write(header)
                 logging.info(f"Created metadata file: {self.metadata_path}")
            else:
                 # Check if header exists and is correct
                 with open(self.metadata_path, "r+", encoding="utf-8") as f:
                      first_line = f.readline()
                      if first_line != header:
                           logging.warning(f"Metadata file {self.metadata_path} has incorrect or missing header. Prepending correct header.")
                           content = f.read() # Read the rest of the file
                           f.seek(0) # Go back to the beginning
                           f.write(header) # Write the correct header
                           f.write(content) # Write the original content back
                           f.truncate() # Remove any trailing old content if file shrunk
        except IOError as e:
             logging.error(f"Error ensuring metadata header for {self.metadata_path}: {e}")


    def get_training_text(self):
        """ Provides curated text or allows user input for training samples. """
        # Example texts - consider expanding or loading from a file
        curated_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "She sells sea shells by the sea shore.",
            "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
            "Pack my box with five dozen liquor jugs.",
            "The five boxing wizards jump quickly.",
            "Peter Piper picked a peck of pickled peppers.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
            "All the world's a stage, and all the men and women merely players.",
            "Ask not what your country can do for you – ask what you can do for your country."
        ]

        print(f"\n--- Record Sample for Character: {self.character_name} ---")
        print("Choose text source for recording:")
        print("1. Provide your own text")
        print("2. Use a random curated text")

        while True:
            choice = input("Enter choice (1 or 2): ")
            if choice == "1":
                return input("Enter the text to record: ").strip() # Strip whitespace
            elif choice == "2":
                selected_text = random.choice(curated_texts)
                print(f"\nPlease read the following text clearly:\n---\n{selected_text}\n---")
                input("Press Enter when ready to record...") # Give user time to prepare
                return selected_text
            else:
                print("Invalid choice. Please enter 1 or 2.")


    def record_training_sample(self, sample_duration=10, sample_rate=22050):
        """ Records a training sample with text and saves it to the character's dataset. """
        text = self.get_training_text()
        if not text:
            print("No text provided. Aborting recording.")
            return

        print(f"\nRecording for {sample_duration} seconds (Sample Rate: {sample_rate} Hz)... Speak clearly!")
        try:
            # Ensure sounddevice is available and working
            # Consider adding a check for available devices if errors occur
            audio_data = sd.rec(int(sample_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait() # Wait for recording to complete
            print("Recording finished.")

            # Save the recording
            # Generate a unique filename (e.g., using timestamp) relative to dataset path
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_base = f"sample_{self.character_name}_{timestamp}.wav" # Include character in filename
            filename_full = os.path.join(self.dataset_path, filename_base)
            print(f"Saving sample to {filename_full}...")
            write_wav(filename_full, sample_rate, audio_data)

            # Append to metadata (ensure proper formatting, e.g., LJSpeech)
            # Use relative path in metadata
            with open(self.metadata_path, "a", encoding="utf-8", newline='') as f:
                writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
                writer.writerow([filename_base, text.strip(), text.strip().lower().replace('.', '').replace(',', '')]) # Write relative path, text, and normalized text

            print(f"Sample and metadata saved successfully for character '{self.character_name}'.")

        except sd.PortAudioError as pae:
             logging.error(f"PortAudio error during recording: {pae}")
             print("Error: Could not record audio. Check your microphone connection and sound device settings.")
             get_audio_device_list() # Show available devices to help debug
        except Exception as e:
            logging.error(f"Error during recording or saving sample: {e}", exc_info=True)
            print(f"An error occurred during recording: {e}")


    def train_voice(self, epochs=100, batch_size=16, learning_rate=0.001):
        """ Triggers the voice cloning training script for the current character with customizable parameters. """
        # Ensure the training script exists
        train_script = "voice_clone_train.py"
        if not os.path.exists(train_script):
            print(f"Error: Training script '{train_script}' not found in the current directory.")
            logging.error(f"Training script '{train_script}' not found.")
            return

        # Check if dataset has enough data (basic check)
        if not os.path.exists(self.metadata_path) or os.path.getsize(self.metadata_path) < 50:  # Arbitrary small size
            print(f"Error: Metadata file for character '{self.character_name}' is missing or seems empty.")
            print(f"Please record or add voice samples first: {self.metadata_path}")
            logging.error(f"Metadata file missing or empty for character '{self.character_name}'.")
            return

        print(f"\n--- Starting Training for Character: {self.character_name} ---")
        print(f"Using training script: '{train_script}'")
        print("This might take a long time depending on your dataset size and hardware.")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Output Path: {self.output_path}")
        print("Ensure your dataset and configuration are correct.")

        # Determine the Python executable to use
        python_executable = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe')
        if not os.path.exists(python_executable):
            python_executable = 'python'  # Fallback to system Python if virtual environment is not found

        # Construct the command with arguments
        command = [
            python_executable,  # Use the virtual environment's Python executable
            train_script,
            '--dataset_path', self.dataset_path,
            '--output_path', self.output_path,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--learning_rate', str(learning_rate),
        ]
        logging.info(f"Running training command: {' '.join(command)}")

        # Execute the training script with character-specific paths
        import subprocess
        try:
            # Run the script and capture output in real-time
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1)

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print training progress to console
            rc = process.poll()  # Get the return code

            if rc == 0:
                print(f"\nTraining script finished successfully for '{self.character_name}'.")
                print(f"Check the '{self.output_path}' directory for trained models and logs.")
                logging.info(f"Training successful for character '{self.character_name}'.")
            else:
                print(f"\nTraining script failed for '{self.character_name}' with return code {rc}.")
                logging.error(f"Training failed for character '{self.character_name}' (return code: {rc}).")

        except FileNotFoundError:
            print("Error: 'python' command not found. Make sure Python is in your system PATH.")
            logging.error("'python' command not found during training attempt.")
        except Exception as e:
            logging.error(f"Error running training script for '{self.character_name}': {e}", exc_info=True)
            print(f"An error occurred while running the training script: {e}")


    def test_trained_voice(self, text, output_file="test_output.wav"):
        """ Tests the trained voice model for the current character. """
        if CoquiTTS is None:
             print("Cannot test trained voice: TTS library not available.")
             logging.error("TTS library not available for testing trained voice.")
             return

        # Define paths based on the character's output directory
        trained_model_path = os.path.join(self.output_path, "best_model.pth") # Common convention
        trained_config_path = os.path.join(self.output_path, "config.json") # Common convention

        if os.path.exists(trained_model_path) and os.path.exists(trained_config_path):
            print(f"\n--- Testing Trained Model for Character: {self.character_name} ---")
            print(f"Model Path: {trained_model_path}")
            print(f"Config Path: {trained_config_path}")
            try:
                # Initialize TTS with the character's trained model paths
                # Ensure GPU usage is handled correctly (e.g., based on availability)
                use_gpu = torch.cuda.is_available()
                logging.info(f"Loading trained model with GPU: {use_gpu}")
                tts_tester = CoquiTTS(model_path=trained_model_path, config_path=trained_config_path, gpu=use_gpu)

                # Define the output path for the test file (can be relative or absolute)
                test_output_path = os.path.join(self.output_path, output_file) # Save test in model dir

                print(f"Generating speech for: '{text[:50]}...'")
                logging.info(f"Generating test speech to: {test_output_path}")
                tts_tester.tts_to_file(text=text, file_path=test_output_path)
                print(f"Generated speech saved to: {test_output_path}")

                # Optionally play the generated audio
                play_choice = input("Play the generated audio? (y/n): ").lower()
                if play_choice == 'y':
                     play_audio(test_output_path)

            except Exception as e:
                 logging.error(f"Error testing trained model for '{self.character_name}': {e}", exc_info=True)
                 print(f"An error occurred during testing: {e}")
        else:
            print(f"\nError: Trained model files not found for character '{self.character_name}' in the output directory.")
            print(f"Expected model: {trained_model_path}")
            print(f"Expected config: {trained_config_path}")
            print("Please ensure training completed successfully and files are in the correct location.")
            logging.error(f"Trained model or config not found for character '{self.character_name}' at {self.output_path}")

    def use_trained_voice(self, text):
        """ Convenience method to generate speech using the character's trained model. """
        # Use a character-specific default filename
        output_filename = f"{self.character_name}_generated_speech.wav"
        self.test_trained_voice(text, output_file=output_filename)

    def provide_voice_data(self, source_file_path):
        """ Adds an existing audio file (MP3 or WAV) to the character's dataset. """
        if not os.path.exists(source_file_path):
            print(f"Error: Source file not found: {source_file_path}")
            logging.error(f"Source file not found for provide_voice_data: {source_file_path}")
            return

        filename = os.path.basename(source_file_path)
        # Ensure target is WAV and potentially add character name prefix
        destination_base = f"{self.character_name}_{os.path.splitext(filename)[0]}.wav"
        destination_full = os.path.join(self.dataset_path, destination_base)

        # --- Conversion Logic (MP3 to WAV) ---
        source_to_copy = None # Path of the file to eventually copy/move

        if filename.lower().endswith(".mp3"):
            print(f"Converting MP3 '{filename}' to WAV for character '{self.character_name}'...")
            try:
                audio = AudioSegment.from_mp3(source_file_path)
                # Export directly to the final destination path
                logging.info(f"Exporting converted WAV to: {destination_full}")
                # Ensure consistent format (e.g., 22050 Hz, 1 channel, 16-bit)
                audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2) # 2 bytes = 16 bit
                audio.export(destination_full, format="wav")
                print(f"Converted and saved to: {destination_full}")
                source_to_copy = None # Already saved at destination
            except Exception as e:
                print(f"Error converting MP3 to WAV: {e}")
                logging.error(f"Error converting MP3 '{source_file_path}' to WAV: {e}", exc_info=True)
                return # Stop processing this file
        elif filename.lower().endswith(".wav"):
             source_to_copy = source_file_path # It's already WAV, copy it
        else:
             print(f"Error: Unsupported file format '{os.path.splitext(filename)[1]}'. Please provide MP3 or WAV.")
             logging.error(f"Unsupported file format provided: {filename}")
             return

        # --- Copy WAV if not converted directly ---
        if source_to_copy:
            try:
                print(f"Copying '{filename}' to dataset folder as '{destination_base}'...")
                logging.info(f"Copying WAV {source_to_copy} to {destination_full}")
                shutil.copy2(source_to_copy, destination_full) # copy2 preserves metadata

                # Optionally, resample/rechannel the copied WAV if needed using ffmpeg or pydub
                # <<< FIX: Correct indentation for the try/except block below >>>
                try:
                    audio = AudioSegment.from_wav(destination_full)
                    if audio.frame_rate != 22050 or audio.channels != 1 or audio.sample_width != 2:
                        print(f"Info: Converting copied WAV {destination_base} to standard format (22050Hz, Mono, 16-bit)...")
                        audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
                        audio.export(destination_full, format="wav") # Overwrite with standard format
                except Exception as format_e: # <<< This was line 1008, now correctly indented
                    print(f"Warning: Could not check/convert format of copied WAV {destination_base}: {format_e}")
                    logging.warning(f"Could not check/convert format of {destination_full}: {format_e}")

            except Exception as e:
                logging.error(f"Error copying file '{source_to_copy}' to '{destination_full}': {e}", exc_info=True)
                print(f"Error copying file: {e}")
                return # Stop processing this file

        # --- Add Placeholder to Metadata ---
        try:
            # Ensure header exists first
            self._ensure_metadata_header()
            # Get transcription from user
            transcription = input("Enter the exact transcription for this audio file: ").strip()
            if not transcription:
                 transcription = "<placeholder_transcription>"

            with open(self.metadata_path, "a", encoding="utf-8", newline='') as f:
                writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
                writer.writerow([destination_base, transcription, transcription.lower().replace('.', '').replace(',', '')]) # Write relative path, text, and normalized text

            print(f"Voice data and transcription added for character '{self.character_name}'.")
            logging.info(f"Added '{destination_base}' to dataset for '{self.character_name}'.")

        except Exception as e:
            logging.error(f"Error updating metadata file {self.metadata_path}: {e}", exc_info=True)
            print(f"Error updating metadata: {e}")
            # Consider removing the added audio file if metadata fails
            # if os.path.exists(destination_full):
            #     os.remove(destination_full)


    @measure_execution_time
    def augment_audio(self, relative_file_path, noise_file="background_noise.mp3"):
        """ Applies random augmentation (pitch, speed, noise) to a WAV file within the character's dataset. """
        full_file_path = os.path.join(self.dataset_path, relative_file_path)
        if not os.path.exists(full_file_path):
             print(f"Error: Audio file for augmentation not found: {full_file_path}")
             logging.error(f"Audio file not found for augmentation: {full_file_path}")
             return None
        if not relative_file_path.lower().endswith(".wav"):
             print("Error: Augmentation currently supports only WAV files.")
             return None

        try:
            print(f"Augmenting audio file: {relative_file_path} for character '{self.character_name}'")
            audio = AudioSegment.from_wav(full_file_path)
            augmented_audio = None
            augmentation_type = "None"

            # Randomly choose an augmentation method
            choice = random.choice(["pitch", "speed", "noise", "none"]) # Add 'none' option

            if choice == "pitch":
                # Pitch shifting (adjust semitones) - pydub's method is basic
                octaves = random.uniform(-0.15, 0.15) # Smaller shift range
                new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
                augmented_audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
                # Resample back to original rate
                augmented_audio = augmented_audio.set_frame_rate(audio.frame_rate)
                augmentation_type = f"Pitch shift ({octaves:.2f} octaves)"

            elif choice == "speed":
                # Time stretching/compression
                speed_factor = random.uniform(0.9, 1.1) # Speed up or slow down slightly
                if abs(speed_factor - 1.0) < 0.01: # Avoid tiny changes
                     choice = "none" # Treat as no change
                     augmented_audio = audio
                     augmentation_type = "None"
                elif speed_factor > 1.0:
                     augmented_audio = speedup(audio, playback_speed=speed_factor)
                     augmentation_type = f"Speed change (x{speed_factor:.2f})"
                else: # Slow down - use librosa for better quality
                     print("Slowing down using librosa time_stretch...")
                     try:
                         y, sr = librosa.load(full_file_path, sr=None)
                         y_slow = librosa.effects.time_stretch(y, rate=speed_factor)
                         # Need to save back to AudioSegment or temp file
                         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                              sf.write(tmp_wav.name, y_slow, sr)
                              augmented_audio = AudioSegment.from_wav(tmp_wav.name)
                         os.remove(tmp_wav.name) # Clean up temp file
                         augmentation_type = f"Speed change (x{speed_factor:.2f})"
                     except Exception as librosa_e:
                          print(f"Librosa time_stretch failed: {librosa_e}. Skipping speed augmentation.")
                          augmented_audio = audio
                          augmentation_type = "Speed (slow down failed)"


            elif choice == "noise":
                # Adding background noise
                # Look for noise file relative to script or in a known location
                noise_file_path = noise_file # Assume it's relative or absolute
                if not os.path.exists(noise_file_path):
                     # Try looking in the script's directory
                     script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.' # Handle interactive use
                     noise_file_path_alt = os.path.join(script_dir, noise_file)
                     if os.path.exists(noise_file_path_alt):
                          noise_file_path = noise_file_path_alt
                     else:
                          print(f"Warning: Background noise file '{noise_file}' not found. Skipping noise addition.")
                          logging.warning(f"Background noise file '{noise_file}' not found.")
                          augmented_audio = audio # No change
                          augmentation_type = "Noise (skipped - file missing)"

                if augmentation_type != "Noise (skipped - file missing)":
                     try:
                         noise = AudioSegment.from_file(noise_file_path)
                         # Ensure noise is long enough, loop if necessary
                         if len(noise) < len(audio):
                              loops = (len(audio) // len(noise)) + 1
                              noise = noise * loops
                         noise = noise[:len(audio)] # Trim to match audio length

                         # Add noise at a lower volume (e.g., -15 to -25 dB relative to audio RMS)
                         noise_level = random.uniform(-25, -15)
                         augmented_audio = audio.overlay(noise + noise_level) # Adjust noise volume relative to 0dBFS
                         augmentation_type = f"Noise addition ('{os.path.basename(noise_file_path)}' at {noise_level:.1f}dB)"
                     except Exception as noise_e:
                          print(f"Error loading or applying noise: {noise_e}")
                          logging.error(f"Error applying noise from {noise_file_path}: {noise_e}")
                          augmented_audio = audio
                          augmentation_type = "Noise (skipped - error)"

            else: # choice == "none"
                 augmented_audio = audio
                 augmentation_type = "None"


            # Save the augmented audio with a new name and update metadata
            if augmentation_type not in ["None", "Speed (slow down failed)", "Noise (skipped - file missing)", "Noise (skipped - error)"]:
                base, ext = os.path.splitext(relative_file_path)
                augmented_filename = f"{base}_aug_{choice}{ext}"
                augmented_path_full = os.path.join(self.dataset_path, augmented_filename)

                print(f"Applied augmentation: {augmentation_type}")
                augmented_audio.export(augmented_path_full, format="wav")
                print(f"Augmented audio saved to: {augmented_filename}")
                logging.info(f"Saved augmented file: {augmented_path_full} (Type: {augmentation_type})")

                # Add entry to metadata for the augmented file (copying original text)
                original_text = None
                try:
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f, delimiter='|')
                        next(reader) # Skip header
                        for row in reader: # Skip empty or malformed rows
                            if not row or len(row) != 2:
                                continue
                            if row[0] == relative_file_path:
                                original_text = row[1]
                                break
                except Exception as e:
                     logging.error(f"Could not read metadata to find text for {relative_file_path}: {e}")

                if original_text:
                    with open(self.metadata_path, "a", encoding="utf-8", newline='') as f:
                        writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
                        writer.writerow([augmented_filename, original_text, original_text.lower().replace('.', '').replace(',', '')]) # Write relative path, text, and normalized text
                    logging.info(f"Added metadata entry for augmented file: {augmented_filename}")
                    return augmented_filename # Return relative path of new file
                else:
                     logging.warning(f"Could not find original text for '{relative_file_path}' in metadata. Augmented file '{augmented_filename}' created but not added to metadata.")
                     return None

            else:
                 print(f"No effective augmentation applied ({augmentation_type}).")
                 return None

        except Exception as e:
            logging.error(f"Error during audio augmentation for {relative_file_path}: {e}", exc_info=True)
            return None


    @measure_execution_time
    def trim_silence(self, relative_file_path, silence_thresh=-40, min_silence_len=300, keep_silence_ms=100):
        """ Trims leading/trailing silence from a WAV file within the character's dataset. Overwrites original file. """
        full_file_path = os.path.join(self.dataset_path, relative_file_path)
        if not os.path.exists(full_file_path):
             print(f"Error: Audio file for trimming not found: {full_file_path}")
             logging.error(f"Audio file not found for trimming: {full_file_path}")
             return False
        if not relative_file_path.lower().endswith(".wav"):
             print("Error: Silence trimming currently supports only WAV files.")
             return False

        try:
            print(f"Trimming silence from: {relative_file_path} (Thresh: {silence_thresh}dB, MinLen: {min_silence_len}ms)")
            audio = AudioSegment.from_wav(full_file_path)
            original_duration = len(audio)

            # Use pydub's silence detection utilities
            # Detect non-silent chunks
            non_silent_chunks = sa.silence.split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence_ms # Keep a bit of silence around chunks
            )

            if not non_silent_chunks:
                 print("Warning: No non-silent parts detected. File might be entirely silent or below threshold.")
                 logging.warning(f"No non-silent parts detected in {relative_file_path}")
                 # Optionally delete the file or leave as is
                 # os.remove(full_file_path)
                 # Remove from metadata? Requires care.
                 return False

            # Concatenate the non-silent chunks
            trimmed_audio = sum(non_silent_chunks) # Pydub allows adding segments

            trimmed_duration = len(trimmed_audio)

            # Overwrite the original file only if trimming occurred and duration changed significantly
            if trimmed_duration < original_duration and (original_duration - trimmed_duration) > 10: # Only save if > 10ms trimmed
                print(f"Trimming applied. Original: {original_duration}ms, Trimmed: {trimmed_duration}ms")
                trimmed_audio.export(full_file_path, format="wav") # Overwrite original
                print(f"Overwrote original file with trimmed version: {relative_file_path}")
                logging.info(f"Trimmed '{relative_file_path}': {original_duration}ms -> {trimmed_duration}ms")
                return True
            elif trimmed_duration >= original_duration :
                 print("No reduction in duration after trimming.")
                 logging.info(f"No reduction in duration for {relative_file_path}")
                 return False
            else:
                 print("No significant silence detected to trim.")
                 logging.info(f"No significant trimming applied to {relative_file_path}")
                 return False

        except Exception as e:
            logging.error(f"Error during silence trimming for {relative_file_path}: {e}", exc_info=True)
            return False


    def validate_metadata(self):
        """ Validates character's metadata: checks file existence, format, and text presence. """
        if not os.path.exists(self.metadata_path):
            print(f"Metadata file not found for character '{self.character_name}': {self.metadata_path}")
            logging.warning(f"Metadata file not found: {self.metadata_path}")
            return False

        print(f"\n--- Validating Metadata for Character: {self.character_name} ---")
        print(f"Metadata file: {self.metadata_path}")

        valid_entries = []
        issues_found = 0
        header = None
        needs_overwrite = False

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                 print("Metadata file is empty.")
                 return True # Empty is valid, technically

            # 1. Check Header
            expected_header = "audio_file|text|normalized_text\n"
            if lines[0] != expected_header:
                 print("  [Issue Line 1] Invalid or missing header. Expected: 'audio_file|text|normalized_text'")
                 # Attempt to fix header later if overwriting
                 header = expected_header # Use correct header for potential rewrite
                 issues_found += 1
                 needs_overwrite = True
            else:
                 header = lines[0]
                 valid_entries.append(header) # Keep valid header

            # 2. Check Data Lines
            processed_filenames = set()
            for i, line in enumerate(lines[1:], 1): # Start from second line (index 1)
                line_num = i + 1
                original_line = line.strip()
                if not original_line:
                     print(f"  [Info Line {line_num}] Skipping empty line.")
                     # issues_found += 1 # Don't count as critical issue, just skip
                     needs_overwrite = True # Need to remove empty lines
                     continue

                parts = original_line.split("|", 2) # Split only twice on the pipe
                if len(parts) != 3:
                    print(f"  [Issue Line {line_num}] Invalid format (missing or extra '|'): {original_line}")
                    issues_found += 1
                    needs_overwrite = True # Need to remove malformed lines
                    continue

                audio_file, text, normalized_text = parts
                audio_file = audio_file.strip() # Remove leading/trailing spaces from filename
                text = text.strip()
                normalized_text = normalized_text.strip()

                # Check for duplicate filenames in metadata
                if audio_file in processed_filenames:
                     print(f"  [Issue Line {line_num}] Duplicate audio file entry found: {audio_file}")
                     issues_found += 1
                     needs_overwrite = True # Need to remove duplicates
                     continue
                processed_filenames.add(audio_file)

                # Check if audio file exists in the character's dataset path
                audio_path = os.path.join(self.dataset_path, audio_file)
                if not os.path.exists(audio_path):
                    print(f"  [Issue Line {line_num}] Audio file not found: {audio_path} (referenced in metadata)")
                    issues_found += 1
                    needs_overwrite = True # Need to remove entries for missing files
                    continue # Skip this entry

                # Check if text is empty or placeholder
                if not text or text.lower() in ["<placeholder_text_please_update>", "<placeholder_transcription>", "<transcription_failed>", "<transcription_error>"]:
                    print(f"  [Warning Line {line_num}] Missing or placeholder text for audio file: {audio_file} ('{text}')")
                    # Don't increment issues_found for this, but keep the entry
                    # User needs to fix this manually

                # If all checks passed for this line (or only warning), add it to valid entries
                valid_entries.append(f"{audio_file}|{text}|{normalized_text}\n") # Reconstruct with stripped parts and newline

            # 3. Summary and Overwrite Option
            if issues_found == 0 and not needs_overwrite:
                print("Metadata validation complete. No critical issues found (check warnings above).")
                return True
            elif issues_found == 0 and needs_overwrite:
                 print("Metadata validation complete. Found minor issues (empty lines/header).")
                 # Ask to fix minor issues
            else:
                print(f"\nMetadata validation complete. Found {issues_found} critical issue(s) requiring file modification.")

            # Ask if overwrite needed
            if needs_overwrite:
                overwrite = input("Overwrite metadata file to fix formatting, remove missing files, and duplicates? (y/n): ").lower()
                if overwrite == 'y':
                    print("Overwriting metadata file...")
                    try:
                        # Backup original file before overwriting
                        backup_path = self.metadata_path + ".bak"
                        shutil.copy2(self.metadata_path, backup_path)
                        print(f"Backup created: {backup_path}")

                        with open(self.metadata_path, "w", encoding="utf-8") as f:
                            f.writelines(valid_entries)
                        print("Metadata file overwritten successfully.")
                        logging.info(f"Overwrote metadata file {self.metadata_path} after validation.")
                        return True # Validation resulted in a clean file
                    except IOError as e:
                         print(f"Error overwriting metadata file: {e}")
                         logging.error(f"Error overwriting metadata file {self.metadata_path}: {e}")
                         return False # Overwrite failed
                    except Exception as e_bak:
                         print(f"Error creating backup or overwriting: {e_bak}")
                         logging.error(f"Error creating backup/overwriting {self.metadata_path}: {e_bak}")
                         return False
                else:
                    print("Metadata file not overwritten.")
                    return False # Validation failed, file not fixed
            else:
                 return True # No critical issues found, no overwrite needed

        except Exception as e:
            logging.error(f"Error during metadata validation for '{self.character_name}': {e}", exc_info=True)
            print(f"An unexpected error occurred during validation: {e}")
            return False


    def check_audio_quality(self, relative_file_path):
        """ Basic audio quality check (clipping, silence) for a file in the character's dataset. """
        full_file_path = os.path.join(self.dataset_path, relative_file_path)
        if not os.path.exists(full_file_path):
             print(f"Error: Audio file for quality check not found: {full_file_path}")
             logging.error(f"Audio file not found for quality check: {full_file_path}")
             return

        try:
            print(f"\n--- Audio Quality Check for: {relative_file_path} (Character: {self.character_name}) ---")
            audio = AudioSegment.from_file(full_file_path) # Use from_file for broader format support initially
            peak_amplitude_dbfs = audio.max_dBFS
            rms_amplitude_dbfs = audio.dBFS # Average loudness

            print(f"  - Duration: {len(audio) / 1000:.2f} seconds")
            print(f"  - Channels: {audio.channels}")
            print(f"  - Sample Rate: {audio.frame_rate} Hz")
            print(f"  - Frame Width (Bytes): {audio.frame_width}") # 1=8bit, 2=16bit, etc.
            print(f"  - RMS Amplitude: {rms_amplitude_dbfs:.2f} dBFS")
            print(f"  - Peak Amplitude: {peak_amplitude_dbfs:.2f} dBFS")

            # Check for potential clipping (peak very close to 0 dBFS)
            if peak_amplitude_dbfs > -0.5: # Threshold can be adjusted
                print("  - WARNING: Potential clipping detected (peak amplitude near 0 dBFS).")
                logging.warning(f"Potential clipping detected in {relative_file_path} (Peak: {peak_amplitude_dbfs:.2f} dBFS)")
            # Check for very low volume (RMS significantly low)
            elif rms_amplitude_dbfs < -45.0: # Threshold can be adjusted
                 print("  - WARNING: Audio level seems very low (RMS < -45 dBFS).")
                 logging.warning(f"Low audio level detected in {relative_file_path} (RMS: {rms_amplitude_dbfs:.2f} dBFS)")
            # Check for silence (using RMS)
            elif rms_amplitude_dbfs < -60.0: # Very low RMS likely indicates silence
                 print("  - WARNING: Audio file appears to be mostly silent (RMS < -60 dBFS).")
                 logging.warning(f"Potential silence detected in {relative_file_path} (RMS: {rms_amplitude_dbfs:.2f} dBFS)")
            else:
                print("  - Basic quality check passed (no obvious clipping or extreme volume issues).")

            # Check consistency (optional but recommended for training)
            if audio.frame_rate != 22050:
                 print(f"  - WARNING: Sample rate is {audio.frame_rate} Hz, expected 22050 Hz for TTS training.")
            if audio.channels != 1:
                 print(f"  - WARNING: Audio is not mono ({audio.channels} channels), expected 1 channel.")
            if audio.frame_width != 2: # Corresponds to 16-bit
                 print(f"  - WARNING: Audio bit depth is not 16-bit ({audio.frame_width*8}-bit), expected 16-bit.")


        except Exception as e:
            logging.error(f"Error during audio quality check for {relative_file_path}: {e}", exc_info=True)
            print(f"An error occurred during quality check: {e}")


    def dataset_statistics(self):
        """ Generates and prints statistics about the character's voice dataset. """
        if not os.path.exists(self.metadata_path):
            print(f"Metadata file not found for character '{self.character_name}': {self.metadata_path}")
            return

        print(f"\n--- Calculating Dataset Statistics for Character: {self.character_name} ---")
        total_duration_ms = 0
        num_samples = 0
        sample_rates = set()
        channels = set()
        bit_depths = set() # Added bit depth check
        valid_files_in_metadata = []
        missing_files = []
        placeholder_texts = 0

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter='|')
                try:
                    _ = next(reader) # Read header
                except StopIteration:
                     print("Metadata file is empty or has no header.")
                     return # Stop if file is empty

                for row in reader: # Skip empty or malformed rows
                    if not row or len(row) != 3:
                        continue

                    audio_file, text, normalized_text = row
                    file_path = os.path.join(self.dataset_path, audio_file)

                    if os.path.exists(file_path):
                        valid_files_in_metadata.append(audio_file)
                        if not text or text.lower() in ["<placeholder_text_please_update>", "<placeholder_transcription>", "<transcription_failed>", "<transcription_error>"]:
                             placeholder_texts += 1
                        try:
                            # Use soundfile for more reliable info reading
                            info = sf.info(file_path)
                            duration_sec = info.duration
                            total_duration_ms += duration_sec * 1000
                            num_samples += 1
                            sample_rates.add(info.samplerate)
                            channels.add(info.channels)
                            # Infer bit depth from subtype (e.g., 'PCM_16' -> 16)
                            subtype = info.subtype
                            if '16' in subtype:
                                bit_depths.add(16)
                            elif '24' in subtype:
                                bit_depths.add(24)
                            elif '32' in subtype:
                                bit_depths.add(32)
                            elif '08' in subtype:
                                bit_depths.add(8)
                            else:
                                bit_depths.add(f"Unknown ({subtype})")

                        except Exception as e:
                            print(f"  Warning: Could not read audio info for file '{audio_file}': {e}")
                            logging.warning(f"Could not read info for {file_path}: {e}")
                    else:
                        missing_files.append(audio_file)

            print("\n--- Dataset Statistics ---")
            print(f"- Character: {self.character_name}")
            print(f"- Metadata File: {self.metadata_path}")
            print(f"- Total Samples in Metadata: {len(valid_files_in_metadata) + len(missing_files)}")
            print(f"- Samples with Audio File Found: {len(valid_files_in_metadata)}")
            if missing_files:
                 print(f"- Samples with Missing Audio Files: {len(missing_files)}")
                 # Optionally list missing files if list is short
                 # if len(missing_files) < 10:
                 #      for mf in missing_files: print(f"    - {mf}")

            if num_samples > 0: # Stats based on files successfully read
                total_duration_sec = total_duration_ms / 1000
                avg_duration_sec = total_duration_sec / num_samples
                print(f"- Total Duration (Readable Files): {total_duration_sec:.2f} seconds ({total_duration_sec/60:.2f} minutes)")
                print(f"- Average Duration (Readable Files): {avg_duration_sec:.2f} seconds")
                print(f"- Detected Sample Rates: {sample_rates}")
                print(f"- Detected Channels: {channels}")
                print(f"- Detected Bit Depths: {bit_depths}")
                print(f"- Samples with Placeholder/Missing Text: {placeholder_texts}")

                # Warnings for inconsistencies (important for training)
                if len(sample_rates) > 1 or any(sr != 22050 for sr in sample_rates):
                     print("  ! WARNING: Inconsistent sample rates detected or rate is not 22050 Hz. Resampling might be needed.")
                if len(channels) > 1 or any(ch != 1 for ch in channels):
                     print("  ! WARNING: Multiple channel counts or non-mono audio detected. Mono audio is typically required.")
                if len(bit_depths) > 1 or any(bd != 16 for bd in bit_depths if isinstance(bd, int)):
                     print("  ! WARNING: Inconsistent bit depths detected or depth is not 16-bit. 16-bit PCM is standard.")
                if placeholder_texts > 0:
                     print(f"  ! WARNING: {placeholder_texts} samples have missing or placeholder transcriptions in the metadata.")
            else:
                print("No valid audio samples found or readable based on the metadata.")
            print("--------------------------")

        except Exception as e:
            logging.error(f"Error calculating dataset statistics for '{self.character_name}': {e}", exc_info=True)
            print(f"An error occurred calculating statistics: {e}")


# --- Original Helper Functions (Sounddevice specific - Keep if needed for direct recording tests) ---
def record_and_play_sd(duration=5, sample_rate=44100, device_index=None):
    """ Records audio and plays it back using sounddevice. """
    logging.info(f"Recording audio (sounddevice) for {duration} seconds...")
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device_index=device_index, dtype="int16")
        sd.wait()
        logging.info("Finished recording. Playing back (sounddevice)...")
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()
        return audio_data
    except Exception as e:
        logging.error(f"Error during sounddevice record/play: {e}")
        return None

def save_wave_file_scipy(audio_data: np.ndarray, filename: str, sample_rate: int = 44100) -> None:
    """ Saves audio data (numpy array) to WAV using scipy.io.wavfile. """
    if audio_data is None:
        logging.error("No audio data provided to save.")
        return
    logging.info(f"Saving audio data (scipy) to WAV file: {filename}")
    try:
        write_wav(filename, sample_rate, audio_data)
        logging.info("WAV file saved successfully (scipy).")
    except Exception as e:
        logging.error(f"Error saving WAV file (scipy): {e}")

def get_audio_device_list():
    """ Prints a list of available audio input/output devices using sounddevice. """
    print("\n--- Available Audio Devices (Sounddevice) ---")
    try:
        devices = sd.query_devices()
        default_input_idx = sd.default.device[0]
        default_output_idx = sd.default.device[1]

        print("Input Devices:")
        found_input = False
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                found_input = True
                default_marker = "*" if i == default_input_idx else " "
                print(f"  {default_marker}[{i}] {device['name']} (Max Channels: {device['max_input_channels']}, Default SR: {int(device['default_samplerate'])})")
        if not found_input:
            print("  No input devices found.")

        print("\nOutput Devices:")
        found_output = False
        for i, device in enumerate(devices):
            if device["max_output_channels"] > 0:
                found_output = True
                default_marker = "*" if i == default_output_idx else " "
                print(f"  {default_marker}[{i}] {device['name']} (Max Channels: {device['max_output_channels']}, Default SR: {int(device['default_samplerate'])})")
        if not found_output:
            print("  No output devices found.")

        print("-------------------------------------------\n")
    except Exception as e:
        print(f"Error querying audio devices: {e}")


# --- Example Usage / Interactive Menu (Character Aware) ---
if __name__ == "__main__":

    print("="*40)
    print(" Voice Tools Interactive Menu")
    print("="*40)

    # --- Character Selection ---
    selected_character = None
    while not selected_character:
        character_input = input("Enter the Character Name to work with: ").strip()
        if character_input:
            # Basic sanitization for safety, VoiceTrainer handles more robustly
            selected_character = re.sub(r'[\\/*?:"<>|]', "_", character_input)
        else:
            print("Character name cannot be empty.")

    # Initialize VoiceTrainer for the selected character
    try:
        trainer = VoiceTrainer(character_name=selected_character)
    except Exception as e:
        print(f"Failed to initialize VoiceTrainer for '{selected_character}': {e}")
        logging.critical(f"Failed to initialize VoiceTrainer for '{selected_character}': {e}", exc_info=True)
        exit(1) # Exit if trainer fails to initialize

    # --- Main Menu Loop ---
    while True:
        print(f"\n--- Main Menu (Character: {trainer.character_name}) ---")
        print("--- Dataset Management ---")
        print("1. Record Training Sample")
        print("2. Add Existing Audio File to Dataset")
        print("3. Validate Dataset Metadata")
        print("4. Show Dataset Statistics")
        print("5. Augment Audio File (Experimental)")
        print("6. Trim Silence from Audio File (Experimental)")
        print("7. Check Audio File Quality (Basic)")
        print("--- Training & Usage ---")
        print("8. Train Voice Model (Runs voice_clone_train.py)")
        print("9. Test Trained Voice Model")
        print("10. Use Trained Voice Model (Generate Speech)")
        print("--- General Tools ---")
        print("11. Transcribe Audio File (STT - Character Agnostic)")
        print("12. Synthesize Speech (Standard TTS - pyttsx3/gTTS)")
        print("13. Synthesize Speech (Pre-trained Cloned Voice - XTTS)")
        print("14. List Audio Devices")
        print("0. Exit")

        choice = input("\nEnter your choice: ")

        try:
            # --- Dataset Management ---
            if choice == "1":
                trainer.record_training_sample()
            elif choice == "2":
                f_path = input("Enter the full path to the audio file (WAV or MP3) to add: ").strip()
                if f_path:
                    trainer.provide_voice_data(f_path)
                else:
                    print("No file path entered.")
            elif choice == "3":
                trainer.validate_metadata()
            elif choice == "4":
                trainer.dataset_statistics()
            elif choice == "5":
                relative_f_path = input("Enter the RELATIVE path (within dataset) of the WAV file to augment (e.g., sample_char_time.wav): ").strip()
                if relative_f_path:
                    trainer.augment_audio(relative_f_path)
                else:
                    print("No file path entered.")
            elif choice == "6":
                relative_f_path = input("Enter the RELATIVE path (within dataset) of the WAV file to trim silence from: ").strip()
                if relative_f_path:
                    trainer.trim_silence(relative_f_path)
                else:
                    print("No file path entered.")
            elif choice == "7":
                relative_f_path = input("Enter the RELATIVE path (within dataset) of the audio file to check quality: ").strip()
                if relative_f_path:
                    trainer.check_audio_quality(relative_f_path)
                else:
                    print("No file path entered.")

            # --- Training & Usage ---
            elif choice == "8":
                trainer.train_voice()
            elif choice == "9":
                text = input("Enter the text to test the trained voice with: ").strip()
                if text:
                     trainer.test_trained_voice(text)
                else:
                     print("No text entered.")
            elif choice == "10":
                text = input("Enter the text to generate speech with using the trained voice: ").strip()
                if text:
                     trainer.use_trained_voice(text)
                else:
                     print("No text entered.")

            # --- General Tools ---
            elif choice == "11":
                stt_file_path = input("Enter the full path to ANY audio file to transcribe: ").strip()
                if not stt_file_path:
                    print("No file path entered.")
                    continue
                if not os.path.exists(stt_file_path):
                     print("File not found.")
                     continue

                print("Select STT Engine:")
                print("  g: Google (Default, Online)")
                print("  s: Sphinx (Offline, Needs Setup)")
                print("  w: Whisper (Offline, Needs Setup, Recommended)")
                print("  v: Vosk (Offline, Needs Model Download)")
                engine_choice = input("Engine (g/s/w/v) [w]: ").lower() or "w" # Default Whisper
                engine_map = {"g": "google", "s": "sphinx", "w": "whisper", "v": "vosk"}
                selected_engine = engine_map.get(engine_choice, "whisper")

                vosk_path = None
                whisper_size = "base"
                if selected_engine == "vosk":
                     vosk_path = input("Enter path to Vosk model directory: ")
                     if not vosk_path or not os.path.isdir(vosk_path):
                          print("Invalid Vosk path provided. Transcription might fail.")
                elif selected_engine == "whisper":
                     whisper_size = input("Enter Whisper model size (tiny, base, small, medium, large) [base]: ") or "base"


                lang_code = input("Enter language code (e.g., en-US, es-ES) [en-US]: ") or "en-US"

                try:
                    stt = SpeechToText(
                        use_microphone=False,
                        audio_file=stt_file_path,
                        engine=selected_engine,
                        vosk_model_path=vosk_path,
                        whisper_model_size=whisper_size
                    )
                    transcript = stt.process_audio(language=lang_code) # Use process_audio for file
                    if transcript:
                        print("\n--- Transcription Result ---")
                        print(transcript)
                        print("--------------------------")
                    else:
                        print("\nTranscription failed or produced no result.")
                except Exception as stt_e:
                     logging.error(f"Error during STT processing: {stt_e}", exc_info=True)
                     print(f"An error occurred during transcription: {stt_e}")

            elif choice == "12":
                text = input("Enter text for standard TTS: ").strip()
                if not text:
                    continue
                use_pyttsx3 = input("Use pyttsx3 engine? (y/n) [y]: ").lower() != 'n'
                lang = input("Enter language code (e.g., en, es) [en]: ") or "en"
                tts = TextToSpeech(use_pyttsx3=use_pyttsx3, lang=lang)
                output_file = input("Save to file? (Enter filename or leave blank to play): ").strip()
                tts.speak(text, play_audio_flag=(not output_file), filename=output_file or None)

            elif choice == "13":
                 if CoquiTTS is None:
                      print("Cloned Voice TTS requires the TTS library. Please install it: pip install TTS")
                      continue
                 text = input("Enter text for pre-trained cloned TTS (e.g., XTTS): ").strip()
                 if not text:
                     continue
                 ref_wav = input("Enter path to reference WAV file for cloning: ").strip()
                 if not ref_wav:
                      print("Reference WAV path needed.")
                      continue
                 if not os.path.exists(ref_wav):
                      print("Reference WAV not found.")
                      continue
                 lang = input("Enter language code (e.g., en, es) [en]: ") or "en"
                 output_file = input("Enter output filename [cloned_output.wav]: ") or "cloned_output.wav"
                 try:
                      # Use the ClonedVoiceTTS class
                      cloner = ClonedVoiceTTS(reference_wavs=ref_wav) # Uses default XTTS model
                      cloner.speak(text, language=lang, output_filename=output_file, play_audio_flag=True)
                 except Exception as clone_e:
                      logging.error(f"Error during voice cloning: {clone_e}", exc_info=True)
                      print(f"An error occurred during cloning: {clone_e}")

            elif choice == "14":
                get_audio_device_list()

            elif choice == "0":
                print("Exiting.")
                break
            else:
                print("Invalid choice. Please try again.")

        except Exception as main_loop_e:
             # Catch unexpected errors in the main loop
             logging.error(f"An error occurred in the main menu: {main_loop_e}", exc_info=True)
             print(f"An unexpected error occurred: {main_loop_e}")
             print("Please check logs for more details.")

