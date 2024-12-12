from manim import *

def intro(scene):
    # Initial zoom out to see the text better
    scene.camera.frame.scale(1.5)
    # Title
    title = Text("Support Vector Machines (SVM)", font_size=60)
    scene.play(Write(title))
    scene.wait(1)
    scene.play(FadeOut(title))
    # Definitions
    definition = VGroup(
        Text("Definition:", font_size=48, weight=BOLD),
        Text("Supervised learning models for classification and regression.", font_size=36),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)  # Arrange within each VGroup
    functionality = VGroup(
        Text("Functionality:", font_size=48, weight=BOLD),
        Text("Find an optimal hyperplane separating data points.", font_size=36),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    application = VGroup(
        Text("Applications:", font_size=48, weight=BOLD),
        Text("Excel in binary classification; handle linear/nonlinear tasks.", font_size=36)
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    definitions = VGroup(definition, functionality, application).arrange(DOWN, aligned_edge=LEFT, buff=0.5) #arrange all VGroups to one big VGroup
    scene.play(AnimationGroup(
        FadeIn(definition),
        FadeIn(functionality),
        FadeIn(application),
        lag_ratio=0.5  # Adjustment for timing between fades
    ))
    scene.wait(2)
    scene.play(FadeOut(definitions))