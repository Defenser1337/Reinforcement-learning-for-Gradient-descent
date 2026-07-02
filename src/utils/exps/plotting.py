import matplotlib.pyplot as plt
import seaborn as sns


def plot_converging_comparasion(result: dict, dim: int, title="_blank_name_", ax=None):
    """Plots the convergence comparison. Returns an Axes instead of calling
    plt.show(), so the function can be reused/tested."""
    sns.set_style("whitegrid")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]

    for i, (name, values) in enumerate(result.items()):
        sns.lineplot(
            x=values[0], y=values[1], label=name,
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2.5, ax=ax,
        )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Optimization step (log)', fontsize=16)
    ax.set_ylabel('Function value (log)', fontsize=16)
    ax.legend(fontsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return ax