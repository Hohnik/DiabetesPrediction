from IPython.display import HTML


def print_model_data(parameters, score, accuracy):

    html_output = """
    <h3>Best Parameters Found</h3>
    <ul>
    """

    for param, value in parameters.items():
        html_output += f"<li>{param}: {value}</li>"

    html_output += f"""
    </ul>
    <h3>Model Performance:</h3>
    <ul>
        <li>Validation Accuracy: <b>{score*100:.2f}%</b></li>
        <li>Test Accuracy: <b>{accuracy*100:.2f}%</b></li>
    </ul>
    """

    return HTML(html_output)