try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "plotting requires matplotlib and numpy. Run: pip install -r requirements.txt"
    ) from exc


def plot_decision_boundary(model, xs, ys, title="Decision Boundary", save_path=None):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    step = 0.02
    grid_x = np.arange(x_min, x_max, step)
    grid_y = np.arange(y_min, y_max, step)

    zz = []
    for y in grid_y:
        row = []
        for x in grid_x:
            row.append(model.predict_proba([x, y]).data)
        zz.append(row)

    plt.figure(figsize=(5, 5))
    plt.contourf(grid_x, grid_y, zz, levels=30, cmap="coolwarm", alpha=0.75)
    xs_np = np.array(xs)
    plt.scatter(
        xs_np[:, 0],
        xs_np[:, 1],
        c=ys,
        cmap="coolwarm",
        edgecolors="black",
        s=80,
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.colorbar(label="Predicted probability")

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    return plt.gcf()
