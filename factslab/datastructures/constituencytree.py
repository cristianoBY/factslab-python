import nltk


class ConstituencyTree(nltk.Tree):
    """A constituency tree

    Extends the nltk.Tree class by adding public methods for get the
    indices of the root, children of a particular node, and parents of
    a particular node.

    In general, the initializer should not be used directly; rather, the
    class method fromstring should be used to build a tree from a parse
    represented as a string using Penn TreeBank annotation conventions
    """

    def __init__(self, node, children=None):
        super().__init__(node, children)

        # precompute positions
        self.positions = self.treepositions()
        self.terminal_indices = [self.leaf_treeposition(i)
                                 for i in range(len(self))]

    def root_idx(self):
        return [()]

    def children_idx(self, idx):
        return [i for i in self.positions
                if (len(idx) + 1) == len(i)]

    def parents_idx(self, idx):
        if len(idx):
            return [idx[:-1]]
        else:
            return []

    def words(self):
        return self.leaves()
