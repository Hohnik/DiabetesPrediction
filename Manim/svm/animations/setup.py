from manim import *

def setup(scene):
    axes = Axes(
        x_range=[0, 5, 1],  # x-axis range
        y_range=[-5, 5, 1],  # y-axis range
        tips=False,
        axis_config={"include_numbers": False,},
        y_axis_config={"scaling": LinearBase()}
    )
    scene.play(Create(axes), run_time=0.5)
    scene.wait(1)