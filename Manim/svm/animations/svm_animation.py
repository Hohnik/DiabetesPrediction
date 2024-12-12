from manim import *
from sklearn.svm import LinearSVC

class SVMExplanation(MovingCameraScene):
    def construct(self):
        # Initial zoom out to see the text better
        self.camera.frame.scale(1.5)
        # Title
        title = Text("Support Vector Machines (SVM)", font_size=60)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

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
        self.play(AnimationGroup(
            FadeIn(definition),
            FadeIn(functionality),
            FadeIn(application),
            lag_ratio=0.5  # Adjustment for timing between fades
        ))
        self.wait(2)
        self.play(FadeOut(definitions))



        # Zoom in again
        self.camera.frame.scale(1)

        # Example with data points and hyperplane (2D data now)
        np.random.seed(0)
        class1 = [np.array([x, y]) for x, y in np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 20)]
        class2 = [np.array([x, y]) for x, y in np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], 20)]

        points1 = VGroup(*[Dot(np.array([x,y,0]), color=BLUE) for x, y in class1])
        points2 = VGroup(*[Dot(np.array([x,y,0]), color=RED) for x, y in class2])

        self.play(FadeIn(points1), FadeIn(points2))

        # Train the SVM
        X = np.array(class1 + class2)
        y = np.array([1] * len(class1) + [-1] * len(class2))
        svm = LinearSVC(random_state=0, C=100)
        svm.fit(X, y)

        # Get the hyperplane parameters from the SVM
        w = svm.coef_[0]
        b = svm.intercept_[0]

        # Create the optimal hyperplane
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1 = np.linspace(x_min, x_max, 100)
        x2 = (-b - w[0] * x1) / w[1]
        optimal_hyperplane = Line(start=np.array([x1[0], x2[0], 0]), end=np.array([x1[-1], x2[-1], 0]), color=YELLOW)

        # Create the margin lines
        margin = 1 / np.linalg.norm(svm.coef_)
        y_down = (-b - w[0] * x1 - 1) / w[1]
        y_up = (-b - w[0] * x1 + 1) / w[1]

        margin1 = Line(start=np.array([x1[0], y_up[0], 0]), end=np.array([x1[-1], y_up[-1], 0]), color=YELLOW, stroke_width=2)
        margin2 = Line(start=np.array([x1[0], y_down[0], 0]), end=np.array([x1[-1], y_down[-1], 0]), color=YELLOW, stroke_width=2)

        self.play(Create(optimal_hyperplane))
        self.play(Create(margin1), Create(margin2))

        margin_group = VGroup(margin1, margin2)

        margin_text = Text("Margin", color=YELLOW, font_size=24).next_to(optimal_hyperplane, UP, buff=0.7)
        self.play(Write(margin_text))

        ## Highlight support vectors
        #support_vectors = VGroup()
        #for point in points1:
        #    projection = optimal_hyperplane.get_projection(point.get_center()) 
        #    distance = np.linalg.norm(point.get_center() - projection)
        #    if distance < 0.6:  # Adjust threshold as needed
        #        support_vectors.add(point.copy().set_color(PURPLE))
        #for point in points2:
        #    projection = optimal_hyperplane.get_projection(point.get_center()) 
        #    distance = np.linalg.norm(point.get_center() - projection)
        #    if distance < 0.6:  # Adjust threshold as needed
        #        support_vectors.add(point.copy().set_color(PURPLE))
        #self.play(AnimationGroup(*[Transform(point, support_vectors[i]) for i, point in enumerate(support_vectors)], lag_ratio=0.2))
        #support_text = Text("Support Vectors", color=PURPLE, font_size=24).next_to(margin_text, UP)
        #self.play(Write(support_text))


        #self.wait(3)
        #self.play(*[FadeOut(mob) for mob in self.mobjects]) #clean scene
#
        ##Non linear example
        #points1 = VGroup(*[Dot(point, color=BLUE) for point in class1]).shift(LEFT*3)
        #points2 = VGroup(*[Dot(point, color=RED) for point in class2]).shift(LEFT*3)
#
        #self.play(FadeIn(points1), FadeIn(points2))
#
        ##Draw a circle around one of the classes
        #circle = Circle(radius=3, color=GREEN).shift(LEFT*3)
        #self.play(Create(circle))
        #nonlinear_text = Text("Non-Linear Separation", color=GREEN, font_size=36).next_to(circle, RIGHT)
        #self.play(Write(nonlinear_text))
        #self.wait(3)