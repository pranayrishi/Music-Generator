import math
import os
import random
import struct
import threading
import time
import tkinter as tk
import wave
from dataclasses import dataclass
from enum import Enum
from tkinter import filedialog, messagebox, ttk
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
    
    def generate_808_kick(self, duration: float = 0.8, pitch: float = 60, variation: float = 0) -> np.ndarray:
        """Generate authentic 808 kick drum with variations"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Add random variation to pitch
        pitch += random.uniform(-variation, variation)
        freq = 440 * (2 ** ((pitch - 69) / 12))
        pitch_env = np.exp(-t * (8 + random.uniform(-2, 2)))
        
        # Generate the 808 sound with random harmonics
        kick = np.sin(2 * np.pi * freq * pitch_env * t)
        kick += random.uniform(0.2, 0.4) * np.sin(2 * np.pi * freq * 2 * pitch_env * t)
        kick += random.uniform(0.05, 0.15) * np.sin(2 * np.pi * freq * 0.5 * pitch_env * t)
        
        # Variable amplitude envelope
        decay_rate = 3 + random.uniform(-1, 1)
        amp_env = np.exp(-t * decay_rate)
        kick *= amp_env
        
        # Random click intensity
        click_intensity = random.uniform(0.2, 0.4)
        click = np.exp(-t * 100) * np.random.normal(0, 0.1, samples) * click_intensity
        kick[:int(samples * 0.01)] += click[:int(samples * 0.01)]
        
        return kick * random.uniform(0.7, 0.9)
    
    def generate_trap_snare(self, duration: float = 0.2, variation: float = 0) -> np.ndarray:
        """Generate trap-style snare with variations"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Variable noise and tone
        noise_intensity = random.uniform(0.2, 0.4)
        noise = np.random.normal(0, noise_intensity, samples)
        
        tone_freq1 = 200 + random.uniform(-50, 50)
        tone_freq2 = 400 + random.uniform(-100, 100)
        tone = random.uniform(0.15, 0.25) * np.sin(2 * np.pi * tone_freq1 * t)
        tone += random.uniform(0.08, 0.12) * np.sin(2 * np.pi * tone_freq2 * t)
        
        # Variable envelope
        decay_rate = 15 + random.uniform(-5, 5)
        env = np.exp(-t * decay_rate)
        
        return (noise + tone) * env * random.uniform(0.5, 0.7)
    
    def generate_hihat(self, duration: float = 0.1, closed: bool = True, variation: float = 0) -> np.ndarray:
        """Generate hi-hat sound with variations"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Variable high frequency content
        noise_intensity = random.uniform(0.15, 0.25)
        noise = np.random.normal(0, noise_intensity, samples)
        
        # Random frequency components
        freqs = [8000, 12000, 16000]
        for freq in freqs:
            freq_var = freq + random.uniform(-1000, 1000)
            noise += random.uniform(0.05, 0.15) * np.sin(2 * np.pi * freq_var * t) * np.random.normal(0, 0.1, samples)
        
        # Variable envelope
        decay_rate = (50 if closed else 15) + random.uniform(-10, 10)
        env = np.exp(-t * decay_rate)
        
        return noise * env * random.uniform(0.3, 0.5)
    
    def generate_saw_wave(self, frequency: float, duration: float, filter_freq: float = None, variation: float = 0) -> np.ndarray:
        """Generate sawtooth wave with variations"""
        frequency += random.uniform(-variation, variation)
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Variable harmonic content
        saw = np.zeros(samples)
        num_harmonics = random.randint(15, 25)
        for n in range(1, num_harmonics):
            harmonic_amp = (1/n) * random.uniform(0.8, 1.2)
            saw += harmonic_amp * np.sin(2 * np.pi * n * frequency * t)
        
        # Apply variable low-pass filter
        if filter_freq:
            filter_freq += random.uniform(-filter_freq*0.3, filter_freq*0.3)
            saw = self.low_pass_filter(saw, filter_freq)
        
        return saw * random.uniform(0.8, 1.0)
    
    def generate_square_wave(self, frequency: float, duration: float, pulse_width: float = 0.5, variation: float = 0) -> np.ndarray:
        """Generate square wave with variable pulse width"""
        frequency += random.uniform(-variation, variation)
        pulse_width += random.uniform(-0.1, 0.1)
        pulse_width = max(0.1, min(0.9, pulse_width))
        
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        square = np.sign(np.sin(2 * np.pi * frequency * t) + (1 - 2 * pulse_width))
        return square * random.uniform(0.25, 0.35)
    
    def generate_distorted_guitar(self, frequency: float, duration: float, distortion: float = 2.0, variation: float = 0) -> np.ndarray:
        """Generate distorted guitar with variations"""
        frequency += random.uniform(-variation, variation)
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Variable harmonic content
        guitar = np.sin(2 * np.pi * frequency * t)
        guitar += random.uniform(0.2, 0.4) * np.sin(2 * np.pi * frequency * 2 * t)
        guitar += random.uniform(0.1, 0.3) * np.sin(2 * np.pi * frequency * 3 * t)
        guitar += random.uniform(0.05, 0.15) * np.sin(2 * np.pi * frequency * 4 * t)
        
        # Variable distortion
        distortion += random.uniform(-0.5, 0.5)
        guitar = np.tanh(guitar * distortion) * random.uniform(0.6, 0.8)
        
        # Variable envelope
        env = self.create_guitar_envelope(samples, variation=0.2)
        return guitar * env
    
    def generate_pad_sound(self, frequency: float, duration: float, variation: float = 0) -> np.ndarray:
        """Generate ambient pad with variations"""
        frequency += random.uniform(-variation, variation)
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Variable detuning
        detune1 = random.uniform(0.005, 0.015)
        detune2 = random.uniform(0.005, 0.015)
        
        pad = np.sin(2 * np.pi * frequency * t)
        pad += random.uniform(0.6, 0.8) * np.sin(2 * np.pi * frequency * (1 + detune1) * t)
        pad += random.uniform(0.4, 0.6) * np.sin(2 * np.pi * frequency * (1 - detune2) * t)
        pad += random.uniform(0.2, 0.4) * np.sin(2 * np.pi * frequency * 2 * t)
        
        # Variable attack
        attack_time = random.uniform(duration * 0.2, duration * 0.4)
        attack_samples = int(attack_time * self.sample_rate)
        envelope = np.ones(samples)
        if attack_samples < samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** random.uniform(1.5, 2.5)
        
        return pad * envelope * random.uniform(0.25, 0.35)
    
    def generate_pluck_sound(self, frequency: float, duration: float, variation: float = 0) -> np.ndarray:
        """Generate plucked string with variations"""
        frequency += random.uniform(-variation, variation)
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        pluck = np.sin(2 * np.pi * frequency * t)
        pluck += random.uniform(0.4, 0.6) * np.sin(2 * np.pi * frequency * 2 * t)
        
        # Variable decay
        decay_rate = 4 + random.uniform(-1, 1)
        env = np.exp(-t * decay_rate)
        
        return pluck * env * random.uniform(0.35, 0.45)
    
    def low_pass_filter(self, signal: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Simple low-pass filter"""
        RC = 1.0 / (cutoff_freq * 2 * np.pi)
        dt = 1.0 / self.sample_rate
        alpha = dt / (RC + dt)
        
        filtered = np.zeros_like(signal)
        filtered[0] = signal[0]
        
        for i in range(1, len(signal)):
            filtered[i] = filtered[i-1] + alpha * (signal[i] - filtered[i-1])
        
        return filtered
    
    def create_guitar_envelope(self, samples: int, variation: float = 0) -> np.ndarray:
        """Create guitar-like envelope with variations"""
        attack_ratio = 0.05 + random.uniform(-variation, variation)
        decay_ratio = 0.3 + random.uniform(-variation, variation)
        
        attack = int(samples * max(0.01, attack_ratio))
        decay = int(samples * max(0.1, decay_ratio))
        sustain = samples - attack - decay
        
        env = np.ones(samples)
        if attack > 0:
            env[:attack] = np.linspace(0, 1, attack)
        if decay > 0 and attack + decay < samples:
            env[attack:attack+decay] = np.linspace(1, 0.6, decay)
        if sustain > 0:
            sustain_decay = random.uniform(3, 7)
            env[attack+decay:] = 0.6 * np.exp(-np.linspace(0, sustain_decay, sustain))
        
        return env

class AdvancedMusicGenerator:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.synth = SoundSynthesis(sample_rate)
        self.setup_genre_data()
    
    def setup_genre_data(self):
        """Setup genre-specific musical data with variations"""
        self.notes = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        
        # Multiple chord progressions for variation
        self.chord_progressions = {
            Genre.PHONK: [
                ['Cm', 'Fm', 'Gm', 'Fm'],
                ['Cm', 'Gm', 'Ab', 'Fm'],
                ['Dm', 'Gm', 'Cm', 'Fm'],
            ],
            Genre.RAP: [
                ['Cm', 'Cm', 'Fm', 'Gm'],
                ['Am', 'Am', 'Dm', 'Em'],
                ['Fm', 'Fm', 'Bb', 'Cm'],
            ],
            Genre.POP: [
                ['C', 'Am', 'F', 'G'],
                ['G', 'Em', 'C', 'D'],
                ['F', 'Dm', 'Bb', 'C'],
            ],
            Genre.ELECTRONIC: [
                ['C', 'F', 'Am', 'G'],
                ['Em', 'C', 'G', 'D'],
                ['Am', 'F', 'C', 'G'],
            ],
            Genre.ROCK: [
                ['C5', 'F5', 'G5', 'F5'],
                ['E5', 'A5', 'B5', 'A5'],
                ['G5', 'C5', 'D5', 'C5'],
            ],
            Genre.AMBIENT: [
                ['Cmaj7', 'Fmaj7', 'Am7', 'G7'],
                ['Dmaj7', 'Gmaj7', 'Bm7', 'A7'],
                ['Fmaj7', 'Bbmaj7', 'Dm7', 'C7'],
            ]
        }
    
    def get_random_progression(self, genre: Genre) -> List[str]:
        """Get a random chord progression for the genre"""
        return random.choice(self.chord_progressions[genre])
    
    def generate_phonk_track(self, duration: float, progress_callback=None) -> np.ndarray:
        """Generate unique phonk music"""
        if progress_callback:
            progress_callback("🔥 Generating dark PHONK atmosphere...")
        
        # Random variations
        bpm = random.randint(135, 145)
        beat_duration = 60 / bpm
        total_beats = int(duration / beat_duration)
        
        track = np.array([])
        progression = self.get_random_progression(Genre.PHONK)
        
        for beat in range(0, total_beats, 16):
            if progress_callback:
                progress = (beat / total_beats) * 100
                progress_callback(f"Building PHONK beat... {progress:.0f}%")
            
            section = np.array([])
            
            # Randomized drum pattern
            kick_pattern = random.choice([
                [0, 6, 10, 14],
                [0, 4, 8, 12],
                [0, 6, 8, 14]
            ])
            
            for i in range(16):
                if i in kick_pattern:
                    pitch_var = random.uniform(-5, 5)
                    kick = self.synth.generate_808_kick(beat_duration, pitch=45 + pitch_var, variation=3)
                    section = self._add_audio(section, kick, i * beat_duration)
                
                if i % 4 == 2:
                    snare = self.synth.generate_trap_snare(beat_duration * 0.3, variation=0.1)
                    section = self._add_audio(section, snare, i * beat_duration)
                
                if i % 2 == 1 and random.random() > 0.3:
                    hihat = self.synth.generate_hihat(beat_duration * 0.1, variation=0.05)
                    section = self._add_audio(section, hihat, i * beat_duration)
            
            # Random bass line
            bass_root = random.choice(['C2', 'D2', 'F2', 'G2'])
            for i in range(4):
                freq = self.notes[bass_root[0]] / 4
                bass = self.synth.generate_saw_wave(freq, beat_duration * 4, filter_freq=150, variation=5)
                bass = np.tanh(bass * random.uniform(2.5, 3.5)) * 0.6
                section = self._add_audio(section, bass, i * beat_duration * 4)
            
            # Random dark melody
            if random.random() > 0.4:
                melody_notes = random.choice([
                    ['C4', 'Eb4', 'F4', 'G4'],
                    ['D4', 'F4', 'G4', 'Bb4'],
                    ['F4', 'Ab4', 'Bb4', 'C5']
                ])
                
                for i in range(8):
                    if random.random() > 0.4:
                        note = random.choice(melody_notes)
                        freq = self.notes[note[0]]
                        melody = self.synth.generate_square_wave(freq, beat_duration * 2, 
                                                               pulse_width=random.uniform(0.2, 0.4), 
                                                               variation=2)
                        melody = self.synth.low_pass_filter(melody, random.uniform(600, 1000))
                        section = self._add_audio(section, melody * 0.3, i * beat_duration * 2)
            
            track = np.concatenate([track, section])
        
        # Apply random effects
        saturation = random.uniform(1.8, 2.2)
        track = np.tanh(track * saturation) * 0.7
        
        return self._normalize_audio(track)
    
    def generate_rap_track(self, duration: float, progress_callback=None) -> np.ndarray:
        """Generate unique rap beat"""
        if progress_callback:
            progress_callback("🎤 Creating RAP foundation...")
        
        bpm = random.randint(80, 90)
        beat_duration = 60 / bpm
        total_beats = int(duration / beat_duration)
        
        track = np.array([])
        
        for beat in range(0, total_beats, 16):
            if progress_callback:
                progress = (beat / total_beats) * 100
                progress_callback(f"Laying down RAP beats... {progress:.0f}%")
            
            section = np.array([])
            
            # Varied kick patterns
            kick_pattern = random.choice([
                [0, 6, 10, 14],
                [0, 4, 8, 12, 14],
                [0, 6, 8, 10, 14]
            ])
            
            for i in range(16):
                if i in kick_pattern:
                    pitch_var = random.uniform(-3, 3)
                    kick = self.synth.generate_808_kick(beat_duration * 1.5, pitch=50 + pitch_var, variation=2)
                    section = self._add_audio(section, kick, i * beat_duration)
                
                if i in [4, 12]:
                    snare = self.synth.generate_trap_snare(beat_duration * 0.4, variation=0.1)
                    section = self._add_audio(section, snare, i * beat_duration)
                
                # Random hi-hat variations
                if random.random() > 0.2:
                    volume = random.uniform(0.4, 0.7) if i % 2 == 0 else random.uniform(0.2, 0.4)
                    hihat = self.synth.generate_hihat(beat_duration * 0.2, variation=0.03) * volume
                    section = self._add_audio(section, hihat, i * beat_duration)
            
            # Random bass line
            bass_notes = random.choice([
                ['C1', 'C1', 'F1', 'G1'],
                ['A1', 'A1', 'D1', 'E1'],
                ['F1', 'F1', 'Bb1', 'C1']
            ])
            
            for i, note in enumerate(bass_notes):
                freq = self.notes[note[0]] / 8
                bass = self.synth.generate_808_kick(beat_duration * 4, pitch=30 + random.uniform(-2, 2))
                section = self._add_audio(section, bass * random.uniform(0.7, 0.9), i * beat_duration * 4)
            
            # Occasional melodic elements
            if beat % 32 == 0 and random.random() > 0.5:
                melody_freq = random.choice([self.notes['C'], self.notes['F'], self.notes['G']]) * 2
                melody = self.synth.generate_pluck_sound(melody_freq, beat_duration * 8, variation=5)
                section = self._add_audio(section, melody * 0.4, 0)
            
            track = np.concatenate([track, section])
        
        return self._normalize_audio(track)
    
    # Similar randomization for other genres...
    def generate_pop_track(self, duration: float, progress_callback=None) -> np.ndarray:
        """Generate unique pop music"""
        if progress_callback:
            progress_callback("✨ Crafting catchy POP melodies...")
        
        bpm = random.randint(115, 125)
        beat_duration = 60 / bpm
        total_beats = int(duration / beat_duration)
        
        track = np.array([])
        progression = self.get_random_progression(Genre.POP)
        
        for beat in range(0, total_beats, 16):
            if progress_callback:
                progress = (beat / total_beats) * 100
                progress_callback(f"Building POP energy... {progress:.0f}%")
            
            section = np.array([])
            
            # Pop drum pattern with variations
            for i in range(16):
                if i % 4 == 0:
                    kick = self.synth.generate_808_kick(beat_duration, pitch=60 + random.uniform(-2, 2))
                    section = self._add_audio(section, kick * random.uniform(0.6, 0.8), i * beat_duration)
                
                if i % 4 == 2:
                    snare = self.synth.generate_trap_snare(beat_duration * 0.3)
                    section = self._add_audio(section, snare * random.uniform(0.7, 0.9), i * beat_duration)
                
                if i % 2 == 1:
                    hihat = self.synth.generate_hihat(beat_duration * 0.15)
                    section = self._add_audio(section, hihat * random.uniform(0.4, 0.6), i * beat_duration)
            
            # Varied bass and melody
            bass_notes = random.choice([progression, 
                                      [c.replace('maj7', '').replace('m7', 'm').replace('7', '') for c in progression]])
            
            for i, chord in enumerate(bass_notes):
                root = chord[0]
                freq = self.notes[root] / 2
                bass = self.synth.generate_saw_wave(freq, beat_duration * 4, 
                                                  filter_freq=300 + random.uniform(-50, 50))
                section = self._add_audio(section, bass * 0.5, i * beat_duration * 4)
            
            # Random catchy melody
            melody_scale = random.choice([
                ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
                ['G', 'A', 'B', 'C', 'D', 'E', 'F']
            ])
            
            for i in range(8):
                note = random.choice(melody_scale)
                octave = random.choice([4, 5])  # Choose octave randomly
                freq = self.notes[note] * (2 ** octave)
                melody = self.synth.generate_square_wave(freq, beat_duration * 2, 
                                                       pulse_width=random.uniform(0.5, 0.7))
                melody = self.synth.low_pass_filter(melody, random.uniform(1800, 2200))
                section = self._add_audio(section, melody * random.uniform(0.3, 0.5), i * beat_duration * 2)
            
            track = np.concatenate([track, section])
        
        return self._normalize_audio(track)
    
    def generate_electronic_track(self, duration: float, progress_callback=None) -> np.ndarray:
        """Generate unique electronic music"""
        if progress_callback:
            progress_callback("🎛️ Synthesizing ELECTRONIC sounds...")
        
        bpm = random.randint(125, 135)
        beat_duration = 60 / bpm
        total_beats = int(duration / beat_duration)
        track = np.array([])
        
        for beat in range(0, total_beats, 32):
            if progress_callback:
                progress = (beat / total_beats) * 100
                progress_callback(f"Creating ELECTRONIC drops... {progress:.0f}%")
            
            section = np.array([])
            
            # Four-on-the-floor with variations
            for i in range(32):
                if i % 4 == 0:
                    kick = self.synth.generate_808_kick(beat_duration, pitch=55 + random.uniform(-3, 3))
                    section = self._add_audio(section, kick * random.uniform(0.7, 0.9), i * beat_duration)
                
                if i % 2 == 1:
                    hihat = self.synth.generate_hihat(beat_duration * 0.1, closed=True)
                    section = self._add_audio(section, hihat * random.uniform(0.3, 0.5), i * beat_duration)
                
                # Build-up effects
                if i >= 24 and i % 1 == 0:
                    hihat = self.synth.generate_hihat(beat_duration * 0.2, closed=False)
                    section = self._add_audio(section, hihat * 0.3, i * beat_duration)
            
            # Dynamic bass with filter sweeps
            for i in range(8):
                freq = self.notes['C'] / 2
                filter_sweep = 200 + i * random.uniform(30, 70)
                bass = self.synth.generate_saw_wave(freq, beat_duration * 4, filter_freq=filter_sweep)
                section = self._add_audio(section, bass * random.uniform(0.5, 0.7), i * beat_duration * 4)
            
            # Evolving lead synth
            for i in range(16):
                base_freq = self.notes['C'] * 2
                freq_mod = base_freq * (1 + i * random.uniform(0.05, 0.15))
                lead = self.synth.generate_saw_wave(freq_mod, beat_duration * 2, 
                                                  filter_freq=1000 + i * random.uniform(100, 300))
                section = self._add_audio(section, lead * random.uniform(0.2, 0.4), i * beat_duration * 2)
            
            track = np.concatenate([track, section])
        
        return self._normalize_audio(track)
    
    def generate_rock_track(self, duration: float, progress_callback=None) -> np.ndarray:
        """Generate unique rock music"""
        if progress_callback:
            progress_callback("🎸 Shredding ROCK guitars...")
        
        bpm = random.randint(135, 145)
        beat_duration = 60 / bpm
        total_beats = int(duration / beat_duration)
        track = np.array([])
        progression = self.get_random_progression(Genre.ROCK)
        
        for beat in range(0, total_beats, 16):
            if progress_callback:
                progress = (beat / total_beats) * 100
                progress_callback(f"Amplifying ROCK power... {progress:.0f}%")
            
            section = np.array([])
            
            # Rock drum pattern
            for i in range(16):
                if i % 4 == 0:
                    kick = self.synth.generate_808_kick(beat_duration, pitch=50 + random.uniform(-2, 2))
                    section = self._add_audio(section, kick * random.uniform(0.9, 1.1), i * beat_duration)
                
                if i % 4 == 2:
                    snare = self.synth.generate_trap_snare(beat_duration * 0.3)
                    section = self._add_audio(section, snare * random.uniform(1.0, 1.3), i * beat_duration)
                
                hihat = self.synth.generate_hihat(beat_duration * 0.1)
                section = self._add_audio(section, hihat * random.uniform(0.5, 0.7), i * beat_duration)
            
            # Power chords with variations
            chord_variations = [
                [['C', 'G'], ['F', 'C'], ['G', 'D'], ['F', 'C']],
                [['E', 'B'], ['A', 'E'], ['B', 'F#'], ['A', 'E']],
                [['G', 'D'], ['C', 'G'], ['D', 'A'], ['C', 'G']]
            ]
            power_chords = random.choice(chord_variations)
            
            for i, chord in enumerate(power_chords):
                for note in chord:
                    freq = self.notes[note]
                    distortion_level = random.uniform(2.5, 3.5)
                    guitar = self.synth.generate_distorted_guitar(freq, beat_duration * 4, 
                                                                distortion=distortion_level, 
                                                                variation=2)
                    section = self._add_audio(section, guitar * random.uniform(0.6, 0.8), i * beat_duration * 4)
            
            track = np.concatenate([track, section])
        
        return self._normalize_audio(track)
    
    def generate_ambient_track(self, duration: float, progress_callback=None) -> np.ndarray:
        """Generate unique ambient music"""
        if progress_callback:
            progress_callback("🌌 Creating AMBIENT atmosphere...")
        
        track = np.array([])
        progression = self.get_random_progression(Genre.AMBIENT)
        
        # Parse chord names to get individual notes
        chord_notes_map = {
            'Cmaj7': ['C', 'E', 'G', 'B'],   'Fmaj7': ['F', 'A', 'C', 'E'],
            'Am7': ['A', 'C', 'E', 'G'],     'G7': ['G', 'B', 'D', 'F'],
            'Dmaj7': ['D', 'F#', 'A', 'C#'], 'Gmaj7': ['G', 'B', 'D', 'F#'],
            'Bm7': ['B', 'D', 'F#', 'A'],    'A7': ['A', 'C#', 'E', 'G'],
            'Bbmaj7': ['Bb', 'D', 'F', 'A'], 'Dm7': ['D', 'F', 'A', 'C'],
            'C7': ['C', 'E', 'G', 'Bb']
        }
        
        section_duration = duration / len(progression)
        
        for i, chord_name in enumerate(progression):
            if progress_callback:
                progress = (i / len(progression)) * 100
                progress_callback(f"Evolving AMBIENT textures... {progress:.0f}%")
            
            chord_notes = chord_notes_map.get(chord_name, ['C', 'E', 'G', 'B'])
            chord_audio = np.array([])
            
            for note in chord_notes:
                if note in self.notes:
                    freq = self.notes[note]
                    # Multiple octaves with random variations
                    pad1 = self.synth.generate_pad_sound(freq, section_duration, variation=2)
                    pad2 = self.synth.generate_pad_sound(freq * 2, section_duration, variation=3) * 0.5
                    pad3 = self.synth.generate_pad_sound(freq / 2, section_duration, variation=1) * 0.3
                    
                    combined_pad = pad1 + pad2 + pad3
                    
                    if len(chord_audio) == 0:
                        chord_audio = combined_pad
                    else:
                        chord_audio += combined_pad
            
            # Add evolving texture
            texture_freq = random.choice([self.notes[n] for n in chord_notes if n in self.notes]) * 4
            texture = self.synth.generate_pad_sound(texture_freq, section_duration, variation=5) * 0.2
            chord_audio += texture
            
            track = np.concatenate([track, chord_audio])
        
        # Apply variable reverb
        delay_time = random.uniform(0.2, 0.4)
        delay_samples = int(delay_time * self.sample_rate)
        delayed = np.pad(track, (delay_samples, 0), mode='constant')[:-delay_samples]
        reverb_amount = random.uniform(0.3, 0.5)
        track = track + reverb_amount * delayed
        
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
    
    def generate_music(self, genre: Genre, duration: float = 120.0, progress_callback=None) -> np.ndarray:
        """Generate unique music for specified genre"""
        # Set random seed for uniqueness
        random.seed()
        
        if genre == Genre.PHONK:
            return self.generate_phonk_track(duration, progress_callback)
        elif genre == Genre.RAP:
            return self.generate_rap_track(duration, progress_callback)
        elif genre == Genre.POP:
            return self.generate_pop_track(duration, progress_callback)
        elif genre == Genre.ELECTRONIC:
            return self.generate_electronic_track(duration, progress_callback)
        elif genre == Genre.ROCK:
            return self.generate_rock_track(duration, progress_callback)
        elif genre == Genre.AMBIENT:
            return self.generate_ambient_track(duration, progress_callback)
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

class MusicGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Advanced Music Generator 🎵")
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a1a')
        
        self.generator = AdvancedMusicGenerator()
        self.current_audio = None
        self.generation_thread = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#1a1a1a')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(title_frame, text="🎵 Advanced Music Generator 🎵", 
                              font=('Arial', 24, 'bold'), fg='#ffffff', bg='#1a1a1a')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Generate unique, authentic music in different genres", 
                                 font=('Arial', 12), fg='#cccccc', bg='#1a1a1a')
        subtitle_label.pack()
        
        # Genre selection
        genre_frame = tk.Frame(self.root, bg='#1a1a1a')
        genre_frame.pack(pady=20)
        
        tk.Label(genre_frame, text="Select Genre:", font=('Arial', 14, 'bold'), 
                fg='#ffffff', bg='#1a1a1a').pack()
        
        # Genre buttons
        button_frame = tk.Frame(genre_frame, bg='#1a1a1a')
        button_frame.pack(pady=10)
        
        genre_colors = {
            Genre.PHONK: '#8B0000',    # Dark red
            Genre.RAP: '#4B0082',      # Indigo
            Genre.POP: '#FF69B4',      # Hot pink
            Genre.ELECTRONIC: '#00CED1',# Dark turquoise
            Genre.ROCK: '#DC143C',     # Crimson
            Genre.AMBIENT: '#9370DB'   # Medium purple
        }
        
        genre_emojis = {
            Genre.PHONK: '🔥', Genre.RAP: '🎤', Genre.POP: '✨',
            Genre.ELECTRONIC: '🎛️', Genre.ROCK: '🎸', Genre.AMBIENT: '🌌'
        }
        
        for i, genre in enumerate(Genre):
            row = i // 3
            col = i % 3
            
            btn = tk.Button(button_frame, 
                           text=f"{genre_emojis[genre]} {genre.value.upper()}", 
                           font=('Arial', 12, 'bold'),
                           bg=genre_colors[genre], fg='white',
                           width=15, height=2,
                           command=lambda g=genre: self.generate_music(g))
            btn.grid(row=row, column=col, padx=10, pady=5)
        
        # Duration setting
        duration_frame = tk.Frame(self.root, bg='#1a1a1a')
        duration_frame.pack(pady=20)
        
        tk.Label(duration_frame, text="Duration (seconds):", font=('Arial', 12), 
                fg='#ffffff', bg='#1a1a1a').pack(side=tk.LEFT)
        
        self.duration_var = tk.StringVar(value="120")
        duration_entry = tk.Entry(duration_frame, textvariable=self.duration_var, 
                                 font=('Arial', 12), width=10)
        duration_entry.pack(side=tk.LEFT, padx=10)
        
        # Progress bar
        self.progress_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.progress_frame.pack(pady=20, fill=tk.X, padx=50)
        
        self.progress_label = tk.Label(self.progress_frame, text="Ready to generate music!", 
                                      font=('Arial', 11), fg='#cccccc', bg='#1a1a1a')
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate', length=400)
        self.progress_bar.pack(pady=5)
        
        # Control buttons
        control_frame = tk.Frame(self.root, bg='#1a1a1a')
        control_frame.pack(pady=20)
        
        self.save_button = tk.Button(control_frame, text="💾 Save Audio", 
                                    font=('Arial', 12), bg='#2E8B57', fg='white',
                                    command=self.save_audio, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = tk.Button(control_frame, text="⏹️ Stop Generation", 
                                    font=('Arial', 12), bg='#DC143C', fg='white',
                                    command=self.stop_generation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Info text
        info_frame = tk.Frame(self.root, bg='#1a1a1a')
        info_frame.pack(pady=20, fill=tk.X, padx=50)
        
        info_text = tk.Text(info_frame, height=8, width=80, font=('Arial', 10),
                           bg='#2a2a2a', fg='#cccccc', wrap=tk.WORD)
        info_text.pack()
        
        info_content = """🎵 UNIQUE MUSIC GENERATION 🎵

Every generation creates completely unique music! Features:

🔥 PHONK: Dark 808s, distorted elements, trap influences
🎤 RAP: Heavy drums, deep bass, authentic hip-hop patterns  
✨ POP: Catchy melodies, bright synths, polished production
🎛️ ELECTRONIC: Sweeping synths, four-on-the-floor, dynamic builds
🎸 ROCK: Distorted guitars, powerful drums, high energy
🌌 AMBIENT: Atmospheric pads, evolving textures, spatial effects

Click any genre button to generate a fresh, unique track!"""
        
        info_text.insert(tk.END, info_content)
        info_text.config(state=tk.DISABLED)
    
    def update_progress(self, message):
        """Update progress label"""
        self.progress_label.config(text=message)
        self.root.update_idletasks()
    
    def generate_music(self, genre: Genre):
        """Generate music in a separate thread"""
        if self.generation_thread and self.generation_thread.is_alive():
            messagebox.showwarning("Generation in Progress", "Please wait for current generation to complete!")
            return
        
        try:
            duration = float(self.duration_var.get())
            if duration <= 0 or duration > 600:
                messagebox.showerror("Invalid Duration", "Please enter a duration between 1 and 600 seconds!")
                return
        except ValueError:
            messagebox.showerror("Invalid Duration", "Please enter a valid number for duration!")
            return
        
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.progress_bar.start()
        
        self.generation_thread = threading.Thread(target=self._generate_music_thread, args=(genre, duration))
        self.generation_thread.daemon = True
        self.generation_thread.start()
    
    def _generate_music_thread(self, genre: Genre, duration: float):
        """Thread function for music generation"""
        try:
            self.current_audio = self.generator.generate_music(genre, duration, self.update_progress)
            
            self.root.after(0, self._generation_complete, genre)
        except Exception as e:
            self.root.after(0, self._generation_error, str(e))
    
    def _generation_complete(self, genre: Genre):
        """Called when generation is complete"""
        self.progress_bar.stop()
        self.update_progress(f"✅ {genre.value.upper()} track generated successfully!")
        self.save_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        messagebox.showinfo("Generation Complete", 
                           f"🎉 Your unique {genre.value} track is ready!\n\nClick 'Save Audio' to save it to your computer.")
    
    def _generation_error(self, error_msg):
        """Called when generation fails"""
        self.progress_bar.stop()
        self.update_progress("❌ Generation failed!")
        self.stop_button.config(state=tk.DISABLED)
        messagebox.showerror("Generation Error", f"An error occurred: {error_msg}")
    
    def stop_generation(self):
        """Stop the current generation"""
        if self.generation_thread and self.generation_thread.is_alive():
            # Note: This is a simple implementation. For proper thread termination,
            # you would need to implement thread-safe cancellation
            self.progress_bar.stop()
            self.update_progress("⏹️ Generation stopped!")
            self.stop_button.config(state=tk.DISABLED)
    
    def save_audio(self):
        """Save the generated audio"""
        if self.current_audio is None:
            messagebox.showwarning("No Audio", "No audio to save! Generate music first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            title="Save Generated Music"
        )
        
        if filename:
            try:
                self.generator.save_audio(self.current_audio, filename)
                messagebox.showinfo("Saved Successfully", f"Music saved as:\n{filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file: {str(e)}")

def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = MusicGeneratorGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
