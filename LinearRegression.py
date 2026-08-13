
import matplotlib.pyplot as plt


def compute_error_for_line_given_points(b, m, points):
    x = points[:, 0]
    y = points[:, 1]
    predictions = m * x + b
    return np.mean((y - predictions) ** 2)


def step_gradient(b_current, m_current, points, learning_rate):
    x = points[:, 0]
    y = points[:, 1]
    N = float(len(points))

    predictions = m_current * x + b_current
    errors = y - predictions

    b_gradient = -(2 / N) * np.sum(errors)
    m_gradient = -(2 / N) * np.sum(x * errors)

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b, new_m


def gradient_descent_runner(points, starting_b, starting_m,
                             learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    error_history = []

    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
        error_history.append(compute_error_for_line_given_points(b, m, points))

    return b, m, error_history


def visualize_results(points_original, x_col_name, y_col_name, b, m,
                       x_mean, x_std, error_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = points_original[:, 0]
    y = points_original[:, 1]

    # --- Left plot: scatter of data + fitted regression line (original units) ---
    ax1.scatter(x, y, alpha=0.4, s=15, label="Data points")

    x_line = np.linspace(x.min(), x.max(), 100)
    x_line_norm = (x_line - x_mean) / x_std
    y_line = m * x_line_norm + b
    ax1.plot(x_line, y_line, color="red", linewidth=2, label="Fitted line")

    ax1.set_xlabel(x_col_name)
    ax1.set_ylabel(y_col_name)
    ax1.set_title(f"{y_col_name} vs {x_col_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Right plot: error (loss) convergence over iterations ---
    ax2.plot(error_history, color="darkorange")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mean Squared Error")
    ax2.set_title("Error Convergence")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("regression_visualization.png", dpi=150)
    plt.show()


def run():
    csv_path = r"C:\Projects\python\Machine Learning\archive\Student_Performance.csv"

    # Column layout in this dataset:
    # Hours Studied, Previous Scores, Extracurricular Activities (Yes/No),
    # Sleep Hours, Sample Question Papers Practiced, Performance Index
    #
    # "Previous Scores" has by far the strongest correlation with the target
    # ("Performance Index"), so we use that pair for the single-variable fit.
    x_col_name = "Previous Scores"
    y_col_name = "Performance Index"
    x_col_idx = 1
    y_col_idx = 5

    raw = np.genfromtxt(
        csv_path,
        delimiter=",",
        skip_header=1,
        usecols=(x_col_idx, y_col_idx),
    )
    points_original = raw[~np.isnan(raw).any(axis=1)]

    # Normalize x so gradient descent converges smoothly regardless of scale
    x_mean = points_original[:, 0].mean()
    x_std = points_original[:, 0].std()
    points = points_original.copy()
    points[:, 0] = (points[:, 0] - x_mean) / x_std

    # Since x is normalized (mean 0, std 1), a much larger learning rate than
    # the original 0.0001 is appropriate here — that value was tuned for raw,
    # unnormalized inputs and would barely move the parameters otherwise.
    learning_rate = 0.1
    initial_b = 0.0
    initial_m = 0.0
    num_iterations = 2000

    print(
        f"Starting gradient descent at b = {initial_b}, "
        f"m = {initial_m}, "
        f"error = {compute_error_for_line_given_points(initial_b, initial_m, points)}"
    )

    b, m, error_history = gradient_descent_runner(
        points, initial_b, initial_m, learning_rate, num_iterations
    )

    print(
        f"Ending point at b = {b}, "
        f"m = {m}, "
        f"error = {compute_error_for_line_given_points(b, m, points)}"
    )
    print(
        f"(normalized) fit: {y_col_name} = {m:.4f} * "
        f"normalized({x_col_name}) + {b:.4f}"
    )

    visualize_results(
        points_original, x_col_name, y_col_name, b, m, x_mean, x_std, error_history
    )


if __name__ == "__main__":
    run()



#csv_path = r"C:\Projects\python\Machine Learning\archive\Student_Performance.csv"
