import matplotlib.pyplot as plt
import mglearn
from imageio import imread
from sklearn.metrics.pairwise import euclidean_distances
import graphviz
from sklearn.tree import export_graphviz

def DrawDecisionTreeBoundaries(model,X,y,x_label="x-axis",y_label="y-axis",eps=None,ax=None,remove_feature_names=True,title=None):
    if ax is None:
        ax = plt.gca()

    if title is None:
        title = "max_depth=%d" % (model.tree_.max_depth)

    if remove_feature_names and hasattr(model, 'feature_names_in_'):
        delattr(model, 'feature_names_in_')  # delete names to avoid warning message

    mglearn.plots.plot_2d_separator(
        model, X.to_numpy(), eps=eps, fill=True, alpha=0.5, ax=ax
    )
    mglearn.discrete_scatter(X.iloc[:, 0], X.iloc[:, 1], y, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def DrawDecisionTreeBoundaries_andTree(
    model, X, y, height=6, width=16, x_label="x-axis", y_label="y-axis", eps=None, remove_feature_names=True
):
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(width, height),
        subplot_kw={"xticks": (), "yticks": ()},
        gridspec_kw={"width_ratios": [1.5, 2]},
    )
    DrawDecisionTreeBoundaries(model, X, y, x_label, y_label, eps, ax[0], remove_feature_names)
    ax[1].imshow(TreeImg(X.columns, model))
    ax[1].set_axis_off()
    plt.show()
def TreeImg(feature_names, tree):
    """For binary classification only"""
    dot = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        class_names=tree.classes_.astype(str),
        impurity=False,
    )
    # adapted from https://stackoverflow.com/questions/44821349/python-graphviz-remove-legend-on-nodes-of-decisiontreeclassifier
    dot = re.sub("(samples = [0-9]+)\\\\n", "", dot)
    dot = re.sub("value", "counts", dot)
    graph = graphviz.Source(dot, format="png")
    fout = "tmp"
    graph.render(fout)
    return imread(fout + ".png")