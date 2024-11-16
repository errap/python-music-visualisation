import numpy as np
import pygame
from pygame import mixer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Initialize Pygame Mixer for audio
pygame.init()
mixer.init()

# Load audio file
audio_file = "ukrainian.mp3"
mixer.music.load(audio_file)
mixer.music.play(-1)  # Play audio in a loop

# Visualization parameters
volhistory = []
multiplier = 10
frame_count = 0
bpm = 125
beat_interval_ms = 60000 / bpm  # 125 BPM to milliseconds per beat

# Subdivisions for dynamic motion (e.g., quarter and eighth notes)
subdivision_factor = 4
subdivision_interval_ms = beat_interval_ms / subdivision_factor

# Set up Matplotlib figure for 9:16 aspect ratio
fig = plt.figure(figsize=(5.4, 9.6), dpi=100)  # Approximately 9:16
ax = fig.add_subplot(111, polar=True)
fig.patch.set_facecolor('black')  # Black background for the canvas

# Particle class for dynamic visual effects
class Particle:
    def __init__(self, angle, radius, speed, color):
        self.angle = angle
        self.radius = radius
        self.speed = speed
        self.color = color

    def move(self):
        self.radius += self.speed

particles = []  # Container for particles

# Functions for effects
def generate_gradient_color(frame_count):
    """Generate a dynamic RGB color gradient."""
    r = (np.sin(frame_count / 50.0) + 1) / 2  # Normalize to [0, 1]
    g = (np.sin(frame_count / 70.0 + np.pi / 3) + 1) / 2
    b = (np.sin(frame_count / 90.0 + 2 * np.pi / 3) + 1) / 2
    return r, g, b

def get_volume(frame_count):
    """Simulate audio volume levels."""
    beat_wave = np.abs(np.sin(2 * np.pi * (frame_count / (beat_interval_ms / 1000))))
    subdivision_wave = np.abs(np.sin(2 * np.pi * (frame_count / (subdivision_interval_ms / 1000))))
    return (beat_wave + subdivision_wave) * 0.5 * multiplier

def get_pulse_scale(frame_count):
    """Generate a pulse scale that keeps the plot large and pulsing in and out."""
    pulse = 1.3 + 0.2 * np.sin(2 * np.pi * (frame_count / (beat_interval_ms / 1000)))
    return pulse

def add_particles(ax, frame_count):
    """Add particles that respond to the rhythm."""
    if frame_count % 5 == 0:  # Add a particle every 5 frames
        particles.append(
            Particle(
                np.random.uniform(0, 2 * np.pi),
                50,
                np.random.uniform(1, 3),
                generate_gradient_color(frame_count),
            )
        )

    for particle in particles:
        particle.move()
        ax.scatter([particle.angle], [particle.radius], color=particle.color, s=10, alpha=0.6)

def add_music_shapes(ax, volume, frame_count):
    """Add reactive shapes such as polygons."""
    size = volume * 10
    rotation = (frame_count % 360) * np.pi / 180
    polygon_angles = np.linspace(0, 2 * np.pi, 6) + rotation
    polygon_radii = np.full(len(polygon_angles), size)
    ax.fill(polygon_angles, polygon_radii, color=generate_gradient_color(frame_count), alpha=0.4)

# Animation update function
def update(frame):
    global frame_count, volhistory
    frame_count += 2  # Faster animation
    ax.clear()
    ax.set_facecolor('black')  # Black background for polar plot

    # Get the audio volume level and pulse scale
    volume = get_volume(frame_count)
    pulse_scale = get_pulse_scale(frame_count)
    volhistory.append(volume)
    if len(volhistory) > 360:
        volhistory.pop(0)

    # Plot the circular waveform with pulse scaling
    angles = np.linspace(0, 2 * np.pi, len(volhistory))
    radii = np.interp(volhistory, [0, max(volhistory)], [50, 400]) * pulse_scale
    ax.plot(angles, radii, color='white', lw=np.random.uniform(1, 3))

    # Add glitch effects
    for _ in range(4):
        glitch_radii = radii + np.random.uniform(-30, 30, len(radii))
        ax.plot(angles, glitch_radii, color=generate_gradient_color(frame_count), alpha=0.3, lw=0.8)

    # Add particles and shapes
    add_particles(ax, frame_count)
    add_music_shapes(ax, volume, frame_count)

    # Style settings
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# Interactive slider for BPM control
ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
bpm_slider = Slider(ax_slider, 'BPM', 60, 200, valinit=125, valstep=1)

def update_bpm(val):
    global bpm, beat_interval_ms
    bpm = bpm_slider.val
    beat_interval_ms = 60000 / bpm

bpm_slider.on_changed(update_bpm)

# Run the animation
ani = FuncAnimation(fig, update, interval=30)
plt.show()

# Stop music when done
mixer.music.stop()
pygame.quit()
