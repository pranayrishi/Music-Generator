import math
import random
import struct
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class Genre(Enum):
    POP = "pop"
    RAP = "rap"
    PHONK = "phonk"
    ELECTRONIC = "electronic"
    AMBIENT = "ambient"
    ROCK = "rock"

@dataclass
class Note:
    frequency: float
    duration: float
    velocity: float = 1.0

class SoundSynthesis:
    """Advanced sound synthesis for realistic instruments"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def generate_808_kick(self, duration: float = 0.8, pitch: float = 60) -> np.ndarray:
        """Generate authentic 808 kick drum"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Fundamental frequency with pitch bend
        freq = 440 * (2 ** ((pitch - 69) / 12))
        pitch_env = np.exp(-t * 8)  # Pitch drops quickly
        
        # Generate the 808 sound
        kick = np.sin(2 * np.pi * freq * pitch_env * t)
        
        # Add harmonics for fatness
        kick += 0.3 * np.sin(2 * np.pi * freq * 2 * pitch_env * t)
        kick += 0.1 * np.sin(2 * np.pi * freq * 0.5 * pitch_env * t)
        
        # Amplitude envelope (punchy attack, long decay)
        amp_env = np.exp(-t * 3)
        kick *= amp_env
        
        # Add some click for attack
        click = np.exp(-t * 100) * np.random.normal(0, 0.1, samples) * 0.3
        kick[:int(samples * 0.01)] += click[:int(samples * 0.01)]
        
        return kick * 0.8
    
    def generate_trap_snare(self, duration: float = 0.2) -> np.ndarray:
        """Generate trap-style snare"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # White noise component
        noise = np.random.normal(0, 0.3, samples)
        
        # Tone component
        tone = 0.2 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 400 * t)
        
        # Envelope
        env = np.exp(-t * 15)
        
        return (noise + tone) * env * 0.6
    
    def generate_hihat(self, duration: float = 0.1, closed: bool = True) -> np.ndarray:
        """Generate hi-hat sound"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # High frequency noise
        noise = np.random.normal(0, 0.2, samples)
        
        # High-pass filter simulation
        for freq in [8000, 12000, 16000]:
            noise += 0.1 * np.sin(2 * np.pi * freq * t) * np.random.normal(0, 0.1, samples)
        
        # Envelope
        decay_rate = 50 if closed else 15
        env = np.exp(-t * decay_rate)
        
        return noise * env * 0.4
    
    def generate_saw_wave(self, frequency: float, duration: float, filter_freq: float = None) -> np.ndarray:
        """Generate sawtooth wave for electronic sounds"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Sawtooth wave using harmonics
        saw = np.zeros(samples)
        for n in range(1, 20):  # 20 harmonics
            saw += (1/n) * np.sin(2 * np.pi * n * frequency * t)
        
        # Apply low-pass filter if specified
        if filter_freq:
            saw = self.low_pass_filter(saw, filter_freq)
        
        return saw
    
    def generate_square_wave(self, frequency: float, duration: float, pulse_width: float = 0.5) -> np.ndarray:
        """Generate square wave with variable pulse width"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Generate square wave
        square = np.sign(np.sin(2 * np.pi * frequency * t) + (1 - 2 * pulse_width))
        
        return square * 0.3
    
    def generate_distorted_guitar(self, frequency: float, duration: float, distortion: float = 2.0) -> np.ndarray:
        """Generate distorted guitar sound"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Base guitar tone (multiple harmonics)
        guitar = np.sin(2 * np.pi * frequency * t)
        guitar += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
        guitar += 0.2 * np.sin(2 * np.pi * frequency * 3 * t)
        guitar += 0.1 * np.sin(2 * np.pi * frequency * 4 * t)
        
        # Add distortion
        guitar = np.tanh(guitar * distortion) * 0.7
        
        # Apply envelope
        env = self.create_guitar_envelope(samples)
        
        return guitar * env
    
    def generate_pad_sound(self, frequency: float, duration: float) -> np.ndarray:
        """Generate ambient pad sound"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Multiple sine waves slightly detuned
        pad = np.sin(2 * np.pi * frequency * t)
        pad += 0.7 * np.sin(2 * np.pi * frequency * 1.01 * t)  # Slightly detuned
        pad += 0.5 * np.sin(2 * np.pi * frequency * 0.99 * t)  # Slightly detuned down
        pad += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)     # Octave
        
        # Slow attack
        attack_time = min(duration * 0.3, 2.0)
        attack_samples = int(attack_time * self.sample_rate)
        envelope = np.ones(samples)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 2
        
        return pad * envelope * 0.3
    
    def generate_pluck_sound(self, frequency: float, duration: float) -> np.ndarray:
        """Generate plucked string sound"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Karplus-Strong algorithm simulation
        pluck = np.sin(2 * np.pi * frequency * t)
        pluck += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
        
        # Quick decay
        env = np.exp(-t * 4)
        
        return pluck * env * 0.4
    
    def low_pass_filter(self, signal: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Simple low-pass filter"""
        # Simple RC filter approximation
        RC = 1.0 / (cutoff_freq * 2 * np.pi)
        dt = 1.0 / self.sample_rate
        alpha = dt / (RC + dt)
        
        filtered = np.zeros_like(signal)
        filtered[0] = signal[0]
        
        for i in range(1, len(signal)):
            filtered[i] = filtered[i-1] + alpha * (signal[i] - filtered[i-1])
        
        return filtered
    
    def create_guitar_envelope(self, samples: int) -> np.ndarray:
        """Create guitar-like envelope"""
        attack = int(samples * 0.05)
        decay = int(samples * 0.3)
        sustain = samples - attack - decay
        
        env = np.ones(samples)
        env[:attack] = np.linspace(0, 1, attack)
        env[attack:attack+decay] = np.linspace(1, 0.6, decay)
        env[attack+decay:] = 0.6 * np.exp(-np.linspace(0, 5, sustain))
        
        return env

class AdvancedMusicGenerator:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.synth = SoundSynthesis(sample_rate)
        self.setup_genre_data()
    
    def setup_genre_data(self):
        """Setup genre-specific musical data"""
        # Note frequencies (C4 scale)
        self.notes = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        
        # Genre-specific scales and progressions
        self.genre_settings = {
            Genre.PHONK: {
                'scale': ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'],  # C minor
                'bpm': 140,
                'chord_prog': ['Cm', 'Fm', 'Gm', 'Fm'],
                'bass_pattern': [1, 0, 0, 1, 0, 1, 0, 0]
            },
            Genre.RAP: {
                'scale': ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'],  # C minor
                'bpm': 85,
                'chord_prog': ['Cm', 'Cm', 'Fm', 'Gm'],
                'bass_pattern': [1, 0, 0, 0, 1, 0, 0, 0]
            },
            Genre.POP: {
                'scale': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],  # C major
                'bpm': 120,
                'chord_prog': ['C', 'Am', 'F', 'G'],
                'bass_pattern': [1, 0, 1, 0, 1, 0, 1, 0]
            },
            Genre.ELECTRONIC: {
                'scale': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],  # C major
                'bpm': 128,
                'chord_prog': ['C', 'F', 'Am', 'G'],
                'bass_pattern': [1, 0, 0, 1, 1, 0, 0, 1]
            },
            Genre.ROCK: {
                'scale': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],  # C major
                'bpm': 140,
                'chord_prog': ['C5', 'F5', 'G5', 'F5'],
                'bass_pattern': [1, 0, 1, 0, 1, 0, 1, 0]
            },
            Genre.AMBIENT: {
                'scale': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],  # C major
                'bpm': 60,
                'chord_prog': ['Cmaj7', 'Fmaj7', 'Am7', 'G7'],
                'bass_pattern': [1, 0, 0, 0, 0, 0, 0, 0]
            }
        }
    
    def generate_phonk_track(self, duration: float) -> np.ndarray:
        """Generate authentic phonk music"""
        print("🔥 Generating PHONK track with dark 808s and distorted elements...")
        
        settings = self.genre_settings[Genre.PHONK]
        beat_duration = 60 / settings['bpm']
        total_beats = int(duration / beat_duration)
        
        track = np.array([])
        
        # Generate dark, heavy pattern
        for beat in range(0, total_beats, 16):  # 16-beat patterns
            section = np.array([])
            
            # Heavy 808 pattern
            for i in range(16):
                if i % 4 == 0:  # Kick on 1 and 3
                    kick = self.synth.generate_808_kick(beat_duration, pitch=45)  # Lower pitch
                    section = self._add_audio(section, kick, i * beat_duration)
                
                if i % 4 == 2:  # Snare on 2 and 4
                    snare = self.synth.generate_trap_snare(beat_duration * 0.3)
                    section = self._add_audio(section, snare, i * beat_duration)
                
                # Hi-hats
                if i % 2 == 1:
                    hihat = self.synth.generate_hihat(beat_duration * 0.1)
                    section = self._add_audio(section, hihat, i * beat_duration)
            
            # Dark bass line
            bass_notes = ['C2', 'C2', 'F2', 'G2']
            for i, note in enumerate(bass_notes):
                freq = self.notes[note[0]] / 4  # Two octaves down
                bass = self.synth.generate_saw_wave(freq, beat_duration * 4, filter_freq=150)
                # Add distortion
                bass = np.tanh(bass * 3) * 0.6
                section = self._add_audio(section, bass, i * beat_duration * 4)
            
            # Dark melody
            melody_notes = ['C4', 'Eb4', 'F4', 'G4', 'Bb4']
            for i in range(8):
                if random.random() > 0.3:  # Sparse melody
                    note = random.choice(melody_notes)
                    freq = self.notes[note[0]]
                    melody = self.synth.generate_square_wave(freq, beat_duration * 2, pulse_width=0.3)
                    melody = self.synth.low_pass_filter(melody, 800)  # Dark filter
                    section = self._add_audio(section, melody * 0.3, i * beat_duration * 2)
            
            track = np.concatenate([track, section])
        
        # Apply phonk-specific effects
        track = np.tanh(track * 2) * 0.7  # Heavy saturation
        return self._normalize_audio(track)
    
    def generate_rap_track(self, duration: float) -> np.ndarray:
        """Generate authentic rap beat"""
        print("🎤 Generating RAP beat with punchy drums and deep bass...")
        
        settings = self.genre_settings[Genre.RAP]
        beat_duration = 60 / settings['bpm']
        total_beats = int(duration / beat_duration)
        
        track = np.array([])
        
        for beat in range(0, total_beats, 16):
            section = np.array([])
            
            # Classic rap drum pattern
            for i in range(16):
                # Kick drum
                if i in [0, 6, 10, 14]:
                    kick = self.synth.generate_808_kick(beat_duration * 1.5, pitch=50)
                    section = self._add_audio(section, kick, i * beat_duration)
                
                # Snare
                if i in [4, 12]:
                    snare = self.synth.generate_trap_snare(beat_duration * 0.4)
                    section = self._add_audio(section, snare, i * beat_duration)
                
                # Hi-hats (16th note pattern)
                if i % 1 == 0:
                    volume = 0.6 if i % 2 == 0 else 0.3
                    hihat = self.synth.generate_hihat(beat_duration * 0.2) * volume
                    section = self._add_audio(section, hihat, i * beat_duration)
            
            # Deep bass line
            bass_notes = ['C1', 'C1', 'F1', 'G1']
            for i, note in enumerate(bass_notes):
                freq = self.notes[note[0]] / 8  # Three octaves down
                bass = self.synth.generate_808_kick(beat_duration * 4, pitch=30)
                section = self._add_audio(section, bass * 0.8, i * beat_duration * 4)
            
            # Simple melodic elements
            if beat % 32 == 0:  # Every other bar
                melody_freq = self.notes['C'] * 2
                melody = self.synth.generate_pluck_sound(melody_freq, beat_duration * 8)
                section = self._add_audio(section, melody * 0.4, 0)
            
            track = np.concatenate([track, section])
        
        return self._normalize_audio(track)
    
    def generate_pop_track(self, duration: float) -> np.ndarray:
        """Generate catchy pop music"""
        print("✨ Generating POP track with bright synths and catchy hooks...")
        
        settings = self.genre_settings[Genre.POP]
        beat_duration = 60 / settings['bpm']
        total_beats = int(duration / beat_duration)
        
        track = np.array([])
        
        for beat in range(0, total_beats, 16):
            section = np.array([])
            
            # Pop drum pattern
            for i in range(16):
                # Kick
                if i % 4 == 0:
                    kick = self.synth.generate_808_kick(beat_duration, pitch=60)
                    section = self._add_audio(section, kick * 0.7, i * beat_duration)
                
                # Snare
                if i % 4 == 2:
                    snare = self.synth.generate_trap_snare(beat_duration * 0.3)
                    section = self._add_audio(section, snare * 0.8, i * beat_duration)
                
                # Hi-hats
                if i % 2 == 1:
                    hihat = self.synth.generate_hihat(beat_duration * 0.15)
                    section = self._add_audio(section, hihat * 0.5, i * beat_duration)
            
            # Bright bass line
            bass_notes = ['C', 'A', 'F', 'G']
            for i, note in enumerate(bass_notes):
                freq = self.notes[note] / 2
                bass = self.synth.generate_saw_wave(freq, beat_duration * 4, filter_freq=300)
                section = self._add_audio(section, bass * 0.5, i * beat_duration * 4)
            
            # Catchy lead melody
            melody_notes = ['C5', 'E5', 'G5', 'A5', 'C6']
            for i in range(8):
                note = melody_notes[i % len(melody_notes)]
                freq = self.notes[note[0]] * (2 ** (int(note[1]) - 4))
                melody = self.synth.generate_square_wave(freq, beat_duration * 2, pulse_width=0.6)
                melody = self.synth.low_pass_filter(melody, 2000)
                section = self._add_audio(section, melody * 0.4, i * beat_duration * 2)
            
            track = np.concatenate([track, section])
        
        return self._normalize_audio(track)
    
    def generate_electronic_track(self, duration: float) -> np.ndarray:
        """Generate electronic dance music"""
        print("🎛️ Generating ELECTRONIC track with synthesizers and drops...")
        
        settings = self.genre_settings[Genre.ELECTRONIC]
        beat_duration = 60 / settings['bpm']
        total_beats = int(duration / beat_duration)
        
        track = np.array([])
        
        for beat in range(0, total_beats, 32):  # Longer sections for builds
            section = np.array([])
            
            # Four-on-the-floor kick pattern
            for i in range(32):
                if i % 4 == 0:
                    kick = self.synth.generate_808_kick(beat_duration, pitch=55)
                    section = self._add_audio(section, kick * 0.8, i * beat_duration)
                
                # Hi-hats on off-beats
                if i % 2 == 1:
                    hihat = self.synth.generate_hihat(beat_duration * 0.1, closed=True)
                    section = self._add_audio(section, hihat * 0.4, i * beat_duration)
                
                # Build-up with open hi-hats
                if i >= 24 and i % 1 == 0:
                    hihat = self.synth.generate_hihat(beat_duration * 0.2, closed=False)
                    section = self._add_audio(section, hihat * 0.3, i * beat_duration)
            
            # Pulsing bass
            for i in range(8):
                freq = self.notes['C'] / 2
                bass = self.synth.generate_saw_wave(freq, beat_duration * 4, filter_freq=200 + i * 50)
                section = self._add_audio(section, bass * 0.6, i * beat_duration * 4)
            
            # Sweeping lead synth
            for i in range(16):
                freq = self.notes['C'] * 2 * (1 + i * 0.1)  # Rising frequency
                lead = self.synth.generate_saw_wave(freq, beat_duration * 2, filter_freq=1000 + i * 200)
                section = self._add_audio(section, lead * 0.3, i * beat_duration * 2)
            
            track = np.concatenate([track, section])
        
        return self._normalize_audio(track)
    
    def generate_rock_track(self, duration: float) -> np.ndarray:
        """Generate rock music with distorted guitars"""
        print("🎸 Generating ROCK track with distorted guitars and powerful drums...")
        
        settings = self.genre_settings[Genre.ROCK]
        beat_duration = 60 / settings['bpm']
        total_beats = int(duration / beat_duration)
        
        track = np.array([])
        
        for beat in range(0, total_beats, 16):
            section = np.array([])
            
            # Rock drum pattern
            for i in range(16):
                # Heavy kick
                if i % 4 == 0:
                    kick = self.synth.generate_808_kick(beat_duration, pitch=50)
                    section = self._add_audio(section, kick * 1.0, i * beat_duration)
                
                # Snare on 2 and 4
                if i % 4 == 2:
                    snare = self.synth.generate_trap_snare(beat_duration * 0.3)
                    section = self._add_audio(section, snare * 1.2, i * beat_duration)
                
                # Constant hi-hats
                hihat = self.synth.generate_hihat(beat_duration * 0.1)
                section = self._add_audio(section, hihat * 0.6, i * beat_duration)
            
            # Distorted power chords
            power_chords = [
                ['C', 'G'],  # C5
                ['F', 'C'],  # F5
                ['G', 'D'],  # G5
                ['F', 'C']   # F5
            ]
            
            for i, chord in enumerate(power_chords):
                for note in chord:
                    freq = self.notes[note]
                    guitar = self.synth.generate_distorted_guitar(freq, beat_duration * 4, distortion=3.0)
                    section = self._add_audio(section, guitar * 0.7, i * beat_duration * 4)
            
            # Bass line following root notes
            bass_notes = ['C', 'F', 'G', 'F']
            for i, note in enumerate(bass_notes):
                freq = self.notes[note] / 2
                bass = self.synth.generate_distorted_guitar(freq, beat_duration * 4, distortion=1.5)
                section = self._add_audio(section, bass * 0.8, i * beat_duration * 4)
            
            track = np.concatenate([track, section])
        
        return self._normalize_audio(track)
    
    def generate_ambient_track(self, duration: float) -> np.ndarray:
        """Generate atmospheric ambient music"""
        print("🌌 Generating AMBIENT track with atmospheric pads and textures...")
        
        settings = self.genre_settings[Genre.AMBIENT]
        beat_duration = 60 / settings['bpm']
        
        track = np.array([])
        
        # Long, evolving pads
        chord_notes = [
            ['C', 'E', 'G', 'B'],   # Cmaj7
            ['F', 'A', 'C', 'E'],   # Fmaj7
            ['A', 'C', 'E', 'G'],   # Am7
            ['G', 'B', 'D', 'F']    # G7
        ]
        
        for i, chord in enumerate(chord_notes * int(duration / 32)):
            chord_audio = np.array([])
            
            for note in chord:
                freq = self.notes[note]
                # Multiple octaves for richness
                pad1 = self.synth.generate_pad_sound(freq, 32)
                pad2 = self.synth.generate_pad_sound(freq * 2, 32)
                pad3 = self.synth.generate_pad_sound(freq / 2, 32)
                
                combined_pad = pad1 + pad2 * 0.5 + pad3 * 0.3
                
                if len(chord_audio) == 0:
                    chord_audio = combined_pad
                else:
                    chord_audio += combined_pad
            
            # Add subtle texture
            texture_freq = random.choice([self.notes[n] for n in chord]) * 4
            texture = self.synth.generate_pad_sound(texture_freq, 32) * 0.2
            chord_audio += texture
            
            track = np.concatenate([track, chord_audio])
        
        # Apply reverb-like effect
        delay_samples = int(0.3 * self.sample_rate)
        delayed = np.pad(track, (delay_samples, 0), mode='constant')[:-delay_samples]
        track = track + 0.4 * delayed
        
        return self._normalize_audio(track)
    
    def _add_audio(self, base_audio: np.ndarray, new_audio: np.ndarray, start_time: float) -> np.ndarray:
        """Add new audio to base audio at specified time"""
        start_sample = int(start_time * self.sample_rate)
        end_sample = start_sample + len(new_audio)
        
        if len(base_audio) < end_sample:
            base_audio = np.pad(base_audio, (0, end_sample - len(base_audio)), mode='constant')
        
        base_audio[start_sample:end_sample] += new_audio
        return base_audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.8
        return audio
    
    def generate_music(self, genre: Genre, duration: float = 120.0) -> np.ndarray:
        """Generate music for specified genre"""
        if genre == Genre.PHONK:
            return self.generate_phonk_track(duration)
        elif genre == Genre.RAP:
            return self.generate_rap_track(duration)
        elif genre == Genre.POP:
            return self.generate_pop_track(duration)
        elif genre == Genre.ELECTRONIC:
            return self.generate_electronic_track(duration)
        elif genre == Genre.ROCK:
            return self.generate_rock_track(duration)
        elif genre == Genre.AMBIENT:
            return self.generate_ambient_track(duration)
        else:
            raise ValueError(f"Unsupported genre: {genre}")
    
    def save_audio(self, audio: np.ndarray, filename: str):
        """Save audio as WAV file"""
        audio_int = (audio * 32767).astype(np.int16)
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int.tobytes())
        
        print(f"✅ Audio saved as {filename}")

def main():
    """Generate authentic music in different genres"""
    generator = AdvancedMusicGenerator()
    
    genres_to_generate = [
        (Genre.PHONK, "authentic_phonk.wav"),
        (Genre.RAP, "authentic_rap.wav"),
        (Genre.POP, "authentic_pop.wav"),
        (Genre.ELECTRONIC, "authentic_electronic.wav"),
        (Genre.ROCK, "authentic_rock.wav"),
        (Genre.AMBIENT, "authentic_ambient.wav")
    ]
    
    duration = 120  # 2 minutes
    
    print("🎵 Advanced Music Generator - Creating Authentic Genre Tracks 🎵")
    print("=" * 60)
    
    for genre, filename in genres_to_generate:
        print(f"\n🎼 Generating {genre.value.upper()} music...")
        print("-" * 40)
        
        audio = generator.generate_music(genre, duration)
        generator.save_audio(audio, filename)
        
        print(f"🎉 {genre.value.capitalize()} track completed!")
    
    print("\n" + "=" * 60)
    print("🎊 All authentic music tracks generated successfully!")
    print("\nGenerated files:")
    for _, filename in genres_to_generate:
        print(f"  🎵 {filename}")
    print("=" * 60)

if __name__ == "__main__":
    main()
