from matplotlib import pyplot as plt, cm
import numpy as np
import itertools


def plot_confusion_matrix(y_truth=None, y_pred=None, cm_matrix=None, title='Confusion matrix', labels=[],
                          cmap=cm.get_cmap("Blues"),
                          cell_round=2, print_out=False, fig_width=1024, fig_height=1024):
    """
    :param List[List[str]] y_truth: list of list of golden upos
    :param List[List[str]] y_pred: list of list of predicted upos
    :param str title: the title to diplay on the plot
    :param matplotlib.colors.LinearSegmentedColormap cmap: the color of the plot 
    :param bool normalize: whether to normalize or not
    :param int cell_round: number of digits after the dot in the displayed numbers
    :param bool print_out: if True prints the matrix on the stdout
    :param int fig_width: width of the figure that will be created
    :param int fig_height: heigth of the figure that will be created
    :return A figure of size (fig_width, fig_height) of the confusion matrix
    :rtype Figure
    """

    """ generate the confusion matrix """
    n_labels = len(labels)
    if cm_matrix is None:
        if y_truth is None or y_pred is None:
            return

        cm_matrix = np.zeros((n_labels, n_labels), dtype=np.int)
        for t_i, p_i in zip(y_truth, y_pred):
            cm_matrix[t_i][p_i] += 1

    """ print the Confusion Matrix on standard output """
    if print_out:
        print title
        print "-" * (len(title) + 4)
        for x in range(0, cm_matrix.shape[0]):
            row = ""
            for y in range(0, cm_matrix.shape[1]):
                value = str(cm_matrix[x][y])
                while len(value) < cell_round + 2:
                    value += " "
                row += value + "\t"
            print row

    """ plot """
    figure = plt.figure(figsize=(fig_width / 100, fig_height / 100), )
    plt.imshow(cm_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_labels)
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm_matrix.max() / 2.
    for i, j in itertools.product(range(cm_matrix.shape[0]), range(cm_matrix.shape[1])):
        plt.text(j, i, cm_matrix[i, j], horizontalalignment="center",
                 color="white" if cm_matrix[i, j] > thresh else "black")

    plt.tight_layout(pad=3)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


if __name__ == "__main__":
    labels_pn = ['Positive', 'Negative']
    labels_hs = ['Ham', 'Spam']
    Y_pred = [0, 1, 0, 1, 0, 1]
    Y_gold = [0, 0, 0, 1, 1, 1]

    conf_matrix_hs = np.matrix([[52, 0], [3, 50]], dtype=int)
    conf_matrix_ps_knn = np.matrix([[287, 21], [22, 228]], dtype=int)
    conf_matrix_ps_sgd = np.matrix([[302, 6], [16, 234]], dtype=int)
    conf_matrix_ps_dt = np.matrix([[305, 3], [20, 230]], dtype=int)

    title_sgd = 'SGD Confusion Matrix'
    title_knn = 'KNN Confusion Matrix'
    title_dt = 'DecisionTree Confusion Matrix'
    title_hs = "Ham/Spam Confusion Matrix"

    fig = plot_confusion_matrix(y_pred=Y_pred, y_truth=Y_gold, cm_matrix=conf_matrix_ps_knn,
                                title=title_knn, labels=labels_pn,
                                fig_height=500, fig_width=500)
    fig.savefig("../Report/imgs/cm_pn_knn.png")

    plt.show()
