import numpy as np
from manim import *
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.5)

# Train an SVM classifier
clf = SVC(kernel="linear")
clf.fit(X, y)

# Extract model parameters
coef = clf.coef_[0]
intercept = clf.intercept_[0]
slope = -coef[0] / coef[1]
intercept_point = -intercept / coef[1]

# Support vectors
support_vectors = clf.support_vectors_



class SVMVisualization(Scene):
    def construct(self):
        # Set up axes
        axes = Axes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            axis_config={}
        ).to_edge(DOWN)

        # Add labels
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        self.play(Create(axes), Write(labels))

        # Plot data points
        points_class_0 = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE)
            for (x, y), label in zip(X, y) if label == 0
        ])
        points_class_1 = VGroup(*[
            Dot(axes.c2p(x, y), color=RED)
            for (x, y), label in zip(X, y) if label == 1
        ])
        self.play(FadeIn(points_class_0, points_class_1))

        # Decision boundary
        def decision_boundary(x):
            return slope * x + intercept_point

        line = axes.plot(decision_boundary, color=YELLOW)
        self.play(Create(line))

        # Margins
        def margin_1_func(x):
            return slope * x + intercept_point + 1 / np.linalg.norm(coef)

        def margin_2_func(x):
            return slope * x + intercept_point - 1 / np.linalg.norm(coef)

        margin_1 = axes.plot(margin_1_func, color=GREEN)
        margin_2 = axes.plot(margin_2_func, color=GREEN)
        self.play(Create(margin_1), Create(margin_2))

        # Highlight support vectors
        support_vector_dots = VGroup(*[
            Dot(axes.c2p(x, y), color=ORANGE).scale(1.5)
            for x, y in support_vectors
        ])
        self.play(FadeIn(support_vector_dots))

        self.wait()
