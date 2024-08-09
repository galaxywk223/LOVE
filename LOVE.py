import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import pygame
import time
import random
from math import sin, cos, pi, log


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


pygame.mixer.init()

audio_file_path = resource_path("Luv letter.mp3")

pygame.mixer.music.load(audio_file_path)
pygame.mixer.music.play(-1)

num_particles = 5000
num_fireworks = 100
num_firework_explosions = 15
num_wandering_particles = 500

fig, ax = plt.subplots(figsize=(20, 16))
fig.patch.set_facecolor("black")

ax.set_xlim(-120, 120)
ax.set_ylim(-100, 100)
ax.set_facecolor("black")
ax.axis("equal")
ax.axis("off")

particles = ax.scatter([], [], color="pink", s=1)

full_text = """
**            **                    **   *************************
**         **                     **  **            **
**      **                       **    **           **
**   **                         **      **          **
** * * **                      **        **         **
**       **                   **************        **
**          **               **             **      **
**             **           **               **  ** **
**                **       **                 ** ** **
"""

text_obj = ax.text(
    0.5,
    0.5,
    "",
    ha="center",
    va="center",
    fontsize=20,
    family="monospace",
    color="white",
    multialignment="left",
    transform=ax.transAxes,
)

fireworks = ax.scatter([], [], color="yellow", s=1)

wandering_particles = ax.scatter([], [], color="yellow", s=1)

text_done = False

firework_data = {
    "x": np.zeros(num_fireworks * num_firework_explosions),
    "y": np.zeros(num_fireworks * num_firework_explosions),
    "dx": np.zeros(num_fireworks * num_firework_explosions),
    "dy": np.zeros(num_fireworks * num_firework_explosions),
    "lifetime": np.zeros(num_fireworks * num_firework_explosions),
    "colors": np.zeros(num_fireworks * num_firework_explosions, dtype=object),
    "origin_x": np.zeros(num_fireworks * num_firework_explosions),
    "origin_y": np.zeros(num_fireworks * num_firework_explosions),
    "sizes": np.zeros(num_fireworks * num_firework_explosions),
}

wandering_data = {
    "x": np.random.uniform(-100, 100, num_wandering_particles),
    "y": np.random.uniform(-100, 100, num_wandering_particles),
    "dx": np.random.uniform(-0.4, 0.4, num_wandering_particles),
    "dy": np.random.uniform(-0.4, 0.4, num_wandering_particles),
    "colors": np.random.choice(
        [
            "yellow",
            "red",
            "green",
            "blue",
            "purple",
            "orange",
            "#FF5733",
            "#FFC300",
            "#33FF57",
            "#3357FF",
            "#FF33FF",
            "#FF6633",
            "#33FFFF",
        ],
        num_wandering_particles,
    ),
}

firework_colors = [
    "#FF5733",
    "#FFC300",
    "#33FF57",
    "#3357FF",
    "#FF33FF",
    "#FF6633",
    "#33FFFF",
    "orange",
    "purple",
    "red",
    "blue",
]

love_text_colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "#FF5733",
    "#FFC300",
    "#33FF57",
    "#3357FF",
    "#FF33FF",
]


def initialize_firework(i):
    start_idx = i * num_fireworks
    end_idx = (i + 1) * num_fireworks
    firework_data["origin_x"][start_idx:end_idx] = np.random.uniform(-100, 100)
    firework_data["origin_y"][start_idx:end_idx] = np.random.uniform(-100, 100)
    firework_data["x"][start_idx:end_idx] = firework_data["origin_x"][start_idx:end_idx]
    firework_data["y"][start_idx:end_idx] = firework_data["origin_y"][start_idx:end_idx]
    firework_data["dx"][start_idx:end_idx] = np.random.uniform(-1.0, 1.0, num_fireworks)
    firework_data["dy"][start_idx:end_idx] = np.random.uniform(-1.0, 1.0, num_fireworks)
    firework_data["lifetime"][start_idx:end_idx] = np.random.uniform(
        30, 60, num_fireworks
    )
    firework_data["colors"][start_idx:end_idx] = np.random.choice(firework_colors)
    firework_data["sizes"][start_idx:end_idx] = np.random.uniform(5, 50, num_fireworks)


for i in range(num_firework_explosions):
    initialize_firework(i)


def heart_function(t, shrink_ratio: float = 3):
    x = 16 * (sin(t) ** 3)
    y = 13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t)
    x *= shrink_ratio
    y *= shrink_ratio
    return int(x), int(y)


def scatter_inside(x, y, beta=0.15):
    ratio_x = -beta * log(random.random())
    ratio_y = -beta * log(random.random())
    dx = ratio_x * (x)
    dy = ratio_y * (y)
    return x - dx, y - dy


def shrink(x, y, ratio):
    force = -1 / (((x) ** 2 + (y) ** 2) ** 0.6)
    dx = ratio * force * (x)
    dy = ratio * force * (y)
    return x - dx, y - dy


def curve(p):
    return 2 * (2 * sin(4 * p)) / (2 * pi)


class Heart:
    def __init__(self, generate_frame=20):
        self._points = set()
        self._edge_diffusion_points = set()
        self._center_diffusion_points = set()
        self.all_points = {}
        self.build(2000)
        self.generate_frame = generate_frame
        for frame in range(generate_frame):
            self.calc(frame)

    def build(self, number):
        for _ in range(number):
            t = random.uniform(0, 2 * pi)
            x, y = heart_function(t)
            self._points.add((x, y))
        for _x, _y in list(self._points):
            for _ in range(3):
                x, y = scatter_inside(_x, _y, 0.05)
                self._edge_diffusion_points.add((x, y))
        point_list = list(self._points)
        for _ in range(4000):
            x, y = random.choice(point_list)
            x, y = scatter_inside(x, y, 0.17)
            self._center_diffusion_points.add((x, y))

    @staticmethod
    def calc_position(x, y, ratio):
        force = 1 / (((x) ** 2 + (y) ** 2) ** 0.520)
        dx = ratio * force * (x) + random.randint(-1, 1)
        dy = ratio * force * (y) + random.randint(-1, 1)
        return x - dx, y - dy

    def calc(self, generate_frame):
        ratio = 10 * curve(generate_frame / 10 * pi)
        halo_radius = int(4 + 6 * (1 + curve(generate_frame / 10 * pi)))
        halo_number = int(3000 + 4000 * abs(curve(generate_frame / 10 * pi) ** 2))
        all_points = []
        heart_halo_point = set()
        for _ in range(halo_number):
            t = random.uniform(0, 2 * pi)
            x, y = heart_function(t, shrink_ratio=3.8)
            x, y = shrink(x, y, halo_radius)
            if (x, y) not in heart_halo_point:
                heart_halo_point.add((x, y))
                x += random.randint(-14, 14)
                y += random.randint(-14, 14)
                size = random.choice((1, 2, 2))
                all_points.append((x, y, size))
        for x, y in self._points:
            x, y = self.calc_position(x, y, ratio)
            size = random.randint(1, 3)
            all_points.append((x, y, size))
        for x, y in self._edge_diffusion_points:
            x, y = self.calc_position(x, y, ratio)
            size = random.randint(1, 2)
            all_points.append((x, y, size))
        for x, y in self._center_diffusion_points:
            x, y = self.calc_position(x, y, ratio)
            size = random.randint(1, 2)
            all_points.append((x, y, size))
        self.all_points[generate_frame] = all_points


heart = Heart()

text_brightness = 0.0
brightening = True
show_love_text = False
love_text_color_index = 0

love_text_obj = ax.text(
    0.5,
    0.5,
    "",
    ha="center",
    va="center",
    fontsize=50,
    family="monospace",
    color=(0, 0, 0),
    transform=ax.transAxes,
)


def update(frame):
    global text_done, text_brightness, brightening, show_love_text, love_text_color_index

    heart_cycle_length = 200

    heart_scale = frame % heart_cycle_length / (heart_cycle_length - 1)

    theta = np.random.uniform(0, 2 * np.pi, num_particles)
    r = np.random.uniform(0, 1, num_particles) ** 0.5
    particle_x = heart_scale * r * 16 * np.sin(theta) ** 3
    particle_y = (
        heart_scale
        * r
        * (
            13 * np.cos(theta)
            - 5 * np.cos(2 * theta)
            - 2 * np.cos(3 * theta)
            - np.cos(4 * theta)
        )
    )
    particles.set_offsets(np.c_[particle_x, particle_y])

    text_cycle_length = len(full_text)
    if not text_done:
        current_length = min(frame * 2, text_cycle_length)
        text_obj.set_text(full_text[:current_length])
        if current_length >= text_cycle_length:
            text_done = True

    if text_done:
        show_love_text = True

    if show_love_text:
        if brightening:
            text_brightness += 0.01
            if text_brightness >= 1.0:
                brightening = False
        else:
            text_brightness -= 0.01
            if text_brightness <= 0.0:
                brightening = True
                love_text_color_index = (love_text_color_index + 1) % len(
                    love_text_colors
                )

        text_brightness = max(0, min(text_brightness, 1.0))

        love_text_obj.set_color(love_text_colors[love_text_color_index])
        love_text_obj.set_alpha(text_brightness)
        love_text_obj.set_text("I Love You!")
        love_text_obj.set_zorder(11)

    if text_done:
        firework_data["x"] += firework_data["dx"]
        firework_data["y"] += firework_data["dy"]
        firework_data["lifetime"] -= 1
        firework_data["sizes"] *= 0.98

        alpha_values = firework_data["lifetime"] / np.max(firework_data["lifetime"])
        alpha_values = np.clip(alpha_values, 0, 1)

        fireworks.set_alpha(alpha_values)

        mask = firework_data["lifetime"] <= 0
        if np.any(mask):
            for i in np.where(mask)[0] // num_fireworks:
                initialize_firework(i)

        fireworks.set_offsets(np.c_[firework_data["x"], firework_data["y"]])
        fireworks.set_color(firework_data["colors"])
        fireworks.set_sizes(firework_data["sizes"])

        wandering_data["x"] += wandering_data["dx"]
        wandering_data["y"] += wandering_data["dy"]

        out_of_bounds_x = (wandering_data["x"] < -100) | (wandering_data["x"] > 100)
        out_of_bounds_y = (wandering_data["y"] < -100) | (wandering_data["y"] > 100)
        wandering_data["dx"][out_of_bounds_x] *= -1
        wandering_data["dy"][out_of_bounds_y] *= -1

        wandering_particles.set_offsets(np.c_[wandering_data["x"], wandering_data["y"]])
        wandering_particles.set_color(wandering_data["colors"])
        wandering_particles.set_sizes(np.full(num_wandering_particles, 5))

    frame_mod = frame % heart.generate_frame
    heart_points = np.array(heart.all_points[frame_mod])
    heart_x, heart_y, heart_size = (
        heart_points[:, 0],
        heart_points[:, 1],
        heart_points[:, 2],
    )
    ax.scatter(heart_x, heart_y, color="pink", s=heart_size)

    text_obj.set_zorder(10)

    return particles, text_obj, fireworks, wandering_particles, love_text_obj


def reset_text(*args):
    global text_done
    text_done = False
    text_obj.set_text("")
    ani.event_source.stop()
    for i in range(len(full_text)):
        update(i)
    ani.event_source.start()


ani = animation.FuncAnimation(
    fig, update, frames=np.arange(0, 400), interval=20, blit=True, repeat=True
)

text_interval = len(full_text) * 10
fig.canvas.new_timer(interval=text_interval).add_callback(reset_text)

plt.show()

while pygame.mixer.music.get_busy():
    time.sleep(1)
