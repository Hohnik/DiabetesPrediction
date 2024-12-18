from manim import *
from sklearn.svm import LinearSVC


def svm2d(scene):
    # Example with data points and hyperplane (2D data now)
    np.random.seed(0)
    class1 = [np.array([x, y]) for x, y in np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 20)]
    class2 = [np.array([x, y]) for x, y in np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], 20)]
    points1 = VGroup(*[Dot(np.array([x,y,0]), color=BLUE) for x, y in class1])
    points2 = VGroup(*[Dot(np.array([x,y,0]), color=RED) for x, y in class2])
    scene.play(FadeIn(points1), FadeIn(points2))

    # Train the SVM
    X = np.array(class1 + class2)
    y = np.array([1] * len(class1) + [-1] * len(class2))
    svm = LinearSVC(random_state=0, C=100)
    svm.fit(X, y)

    # Get the hyperplane parameters from the SVM
    w = svm.coef_[0]
    b = svm.intercept_[0]

    # Draw suboptimal hyperplane
    hyperplane = Line(start=LEFT * 3, end=RIGHT * 3, color=GREEN)
    scene.play(Create(hyperplane))

    # Create the optimal hyperplane
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1 = np.linspace(x_min, x_max, 100)
    x2 = (-b - w[0] * x1) / w[1]

    # Draw optimal hyperplane
    optimal_hyperplane = Line(start=np.array([x1[0], x2[0], 0]), end=np.array([x1[-1], x2[-1], 0]), color=YELLOW)
    scene.play(Transform(hyperplane, optimal_hyperplane), run_time=2)

    # Create the margin lines
    margin = 1 / np.linalg.norm(svm.coef_)
    y_down = (-b - w[0] * x1 - 1) / w[1]
    y_up = (-b - w[0] * x1 + 1) / w[1]
    margin1 = DashedLine(start=np.array([x1[0], y_up[0], 0]), end=np.array([x1[-1], y_up[-1], 0]), color=YELLOW, stroke_width=2, dash_length=0.3)
    margin2 = DashedLine(start=np.array([x1[0], y_down[0], 0]), end=np.array([x1[-1], y_down[-1], 0]), color=YELLOW, stroke_width=2, dash_length=0.3)
    margin_text = Text("")

    scene.play(Create(optimal_hyperplane))
    scene.play(Create(margin1), Create(margin2))
