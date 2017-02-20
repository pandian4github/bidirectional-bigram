import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_perplexity(x_axis, x_ticks, y_axis1, y_axis2, x_label, y_label, title, axis, file_name):

    print y_axis1
    print y_axis2

    plt.xticks(x_axis, x_ticks)
    plt.plot(x_axis, y_axis1, 'r-o', label='forward')
    plt.plot(x_axis, y_axis2, 'b-o', label='backward')

    red_forward_patch = mpatches.Patch(color='red', label='Forward model', linestyle='solid', linewidth=0.1)
    blue_backward_patch = mpatches.Patch(color='blue', label='Backward model', linestyle='solid', linewidth=0.1)

    lgd = plt.legend(handles=[red_forward_patch, blue_backward_patch], loc='upper right', bbox_to_anchor=(1.4, 1.0))

    plt.grid(True)
    plt.axis(axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    fig = plt.figure(1)
    fig.savefig(file_name, dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.close()


def plot_word_perplexity(x_axis, y_axis1, y_axis2, y_axis3, x_label, y_label, title, axis, file_name):

    plt.plot(x_axis, y_axis1, 'r-o', label='forward')
    plt.plot(x_axis, y_axis2, 'b-o', label='backward')
    plt.plot(x_axis, y_axis3, 'g-o', label='bidirectional')

    red_forward_patch = mpatches.Patch(color='red', label='Forward model', linestyle='solid', linewidth=0.1)
    blue_backward_patch = mpatches.Patch(color='blue', label='Backward model', linestyle='solid', linewidth=0.1)
    green_backward_patch = mpatches.Patch(color='green', label='Bidirectional model', linestyle='solid', linewidth=0.1)

    lgd = plt.legend(handles=[red_forward_patch, blue_backward_patch, green_backward_patch], loc='upper right', bbox_to_anchor=(1.4, 1.0))

    plt.grid(True)
    plt.axis(axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    fig = plt.figure(1)
    fig.savefig(file_name, dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.close()


def main(type):
    x_axis = ['atis', 'wsj', 'brown']
    x_axis = [1, 2, 3]
    if type == 'training':
        # Plotting perplexities
        forward = [7.043192013, 74.26799183, 93.5192772]
        backward = [9.012936244, 74.26778753, 93.50913084]
        x_label = 'Corpus'
        y_label = 'Perplexity'
        title = 'Perplexity for different corpus (' + type + ')'
        file_name = 'plot_perplexity_' + type + '.png'
        axis = [0, 4, 0, 100]
        x_ticks = ['atis', 'wsj', 'brown']
        plot_perplexity(x_axis, x_ticks, forward, backward, x_label, y_label, title, axis, file_name)
    elif type == 'test':
        training_perplexities = [62.96085561829097, 56.88377096390624, 53.34896025157767, 50.99519973381952, 49.34396255061554, 48.171440122996934, 47.357435906384424, 46.83304816334652, 46.558773871482046, 46.514445090631405, 46.69472851118176, 47.10790714145462, 47.777370420863555, 48.74644697298152, 50.08894672537915, 51.93183374713426, 54.508918810983154, 58.31449953667728, 64.73468201202915]
        test_perplexities = [177.08946258375565, 157.53862505431553, 146.58734764028338, 139.4474228803418, 134.504559815185, 131.02427658662242, 128.61976032547457, 127.07234618406198, 126.25784195756114, 126.1131573880389, 126.62152989276237, 127.80855365063704, 129.74707455049523, 132.57312887103024, 136.5210838584855, 142.00041921756323, 149.7814466703079, 161.54180534599658, 182.20253028784037]
        axis = [0, 1, 30, 200]
        corpus = 'wsj'
    else:
        print 'Invalid type!'


main('training')
# main('wsj')
# main('brown')