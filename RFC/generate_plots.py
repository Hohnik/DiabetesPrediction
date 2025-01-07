from dtreeviz import model
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from Transformer import ReplaceZeroWithMean

DATAPATH = "../Data"
MODELPATH = "../Data/Models/RFC"

data = pd.read_csv(f"{DATAPATH}/diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=1)
imputer = ReplaceZeroWithMean(["Glucose", "BloodPressure", "SkinThickness", "BMI"])
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


RFC_URL = "../Data/Models/RFC"
SVM_URL = "../Data/Models/SVM"
models = {
    "RFC": f"{RFC_URL}/RFC_no_hyper.pkl",
    "RFC_hyper": f"{RFC_URL}/RFC_hyper.pkl",
    "RFC_hyper_cv": f"{RFC_URL}/RFC_hyper_cv.pkl",
    "RFC_hyper_cv_tuned": f"{RFC_URL}/RFC_hyper_cv_tuned.pkl",
    # "SVM": f"{SVM_URL}/SVM_no_hyper.pkl",
    # "SVM_para_grid": f"{SVM_URL}/SVM_para_grid.pkl",
    # "SVM_para_sampl": f"{SVM_URL}/SVM_para_sample.pkl",
}


def render_tree_image(model_url, output_url):
    rfc = joblib.load(model_url)
    tree = rfc.estimators_[0]

    viz_model = model(
        tree,
        X_train=X_train,
        y_train=y_train,
        feature_names=X_train.columns,
        target_name="Diabetes",
        class_names=["No", "Yes"],
        tree_index=0,
    )

    tree_url = viz_model.view().save_svg()

    with open(tree_url, "rb") as tmp:
        with open(output_url, "wb") as svg:
            svg.write(tmp.read())


render_tree_image(models["RFC"], "../Data/RFC.svg")
render_tree_image(models["RFC_hyper"], "../Data/RFC_hyper.svg")
render_tree_image(models["RFC_hyper_cv"], "../Data/RFC_hyper_cv.svg")
render_tree_image(models["RFC_hyper_cv_tuned"], "../Data/RFC_hyper_cv_tuned.svg")
