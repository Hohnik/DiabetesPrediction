from manim import *
from animations.intro import intro
from animations.svm2d import svm2d
from animations.setup import setup

class main(MovingCameraScene):
    def construct(self):
        intro(self)
        #setup(self)
        svm2d(self)

