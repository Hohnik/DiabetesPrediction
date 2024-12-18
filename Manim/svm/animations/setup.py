from manim import *

#def setup(scene):
#    axes = Axes(
#        x_range=[0, 5, 1],  # x-axis range
#        y_range=[-5, 5, 1],  # y-axis range
#        tips=False,
#        axis_config={"include_numbers": False,},
#        y_axis_config={"scaling": LinearBase()}
#    )
#    scene.play(Create(axes), run_time=0.5)
#    scene.wait(1)
def setup(scene):
    # Create x-axis (horizontal line)
    x_axis = Line(
        start=np.array([0, -5, 0]),  # Start at y=-5
        end=np.array([5, -5, 0]),    # End at x=5, y=-5
        color=WHITE
    )
    
    # Create y-axis (vertical line)
    y_axis = Line(
        start=np.array([0, -5, 0]),  # Start at y=-5
        end=np.array([0, 5, 0]),     # End at y=5
        color=WHITE
    )
    
    # Group both axes
    axes = VGroup(x_axis, y_axis)
    
    axes.to_corner(DOWN + LEFT)
    
    scene.play(Create(axes), run_time=0.5)
    scene.wait(1)