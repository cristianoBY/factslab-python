import nltk


class DependencyTree(nltk.Tree):
    """A dependency tree
        self.sentence contains the sentence corresponding to structure
        in list form. Assign it after initialistaion
        self.tokens maybe used later.
    """

    def __init__(self, node, children=None):
        super().__init__(node, children)

        # precompute positions
        self.positions = self.treepositions()
        self.sentence = []
        self.tokens = []

    def root_idx(self):
        return [()]

    def children_idx(self, idx):
        if idx != ():
            return [i for i in self.positions
                if (len(idx) + 1) == len(i) and i[0] == idx[0]]
        else:
            return [i for i in self.positions
                if (len(idx) + 1) == len(i)]

    def parents_idx(self, idx):
        if len(idx):
            return [idx[:-1]]
        else:
            return []

    def words(self):
        return self.sentence

    def word_index(self, index):
        if isinstance(self[index], str):
            return self.sentence.index(self[index])
        else:
            return self.sentence.index(self[index].label())
