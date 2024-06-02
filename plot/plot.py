from matplotlib import pyplot as plt


def plot_with_text(x, y, title, x_label, y_label, x_point, y_point, save_file):
    plt.plot(x, y, color='blue', linestyle='dashed', markerfacecolor='red', markersize=10)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x_point, y_point, color='green', s=100)
    plt.text(x_point, y_point, f'{x_label}={x_point}, 'f'{y_label}={y_point:.2f}', fontsize=12,
             verticalalignment='bottom')
    plt.savefig(save_file)
