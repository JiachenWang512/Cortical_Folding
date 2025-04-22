import matplotlib.pyplot as plt


def plot_pattern(pattern, title=None):
    fig, ax = plt.subplots()
    cax = ax.imshow(pattern, cmap='viridis', origin='lower')
    ax.set_title(title or '')
    fig.colorbar(cax, ax=ax)
    return fig, ax