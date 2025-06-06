import re
import graphviz
from sklearn.tree import export_graphviz


def display_tree(feature_names, tree):
    """ 
    Will create a decision tree visualization showing the features used on each split. 
    
    Parameters
    ----------
    feature_names : list
        The list of the features used in the decision tree
    tree : sklearn.tree._classes.DecisionTreeClassifier
        The decision tree model we are using for prediction
        
    Returns
    -------
    graphviz.files.Source 
        The decision tree visualization 
        
    Examples
    --------
    >>> display_tree(X.columns, model])
    
    """
    dot = export_graphviz(tree, out_file=None, feature_names=feature_names, class_names=tree.classes_.astype(str), impurity=False)
    # adapted from https://stackoverflow.com/questions/44821349/python-graphviz-remove-legend-on-nodes-of-decisiontreeclassifier
    #dot = re.sub('(\\\\nsamples = [0-9]+)(\\\\nvalue = \[[0-9]+, [0-9]+\])(\\\\nclass = [A-Za-z0-9]+)', '', dot)
    dot = re.sub('(\\\\nsamples = [0-9]+)(\\\\nvalue = \[[0-9]+, [0-9]+\])', '', dot)
    dot = re.sub(     '(samples = [0-9]+)(\\\\nvalue = \[[0-9]+, [0-9]+\])\\\\n', '', dot)
    
    return graphviz.Source(dot)