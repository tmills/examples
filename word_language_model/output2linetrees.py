#!/usr/bin/env python

import sys

class BinaryTree():
    def __init__(self, cat, lc=None, rc=None, parent=None):
        self.cat = cat
        self.lc = lc
        self.rc = rc
        self.parent = parent

    def __str__(self):
        if self.rc is None:
            return '(%s %s)' % (self.cat, self.lc)
        else:
            return '(%s %s %s)' % (self.cat, str(self.lc), str(self.rc))

class PreTerminal():
    def __init__(self, token):
        self.cat = 'PT'
        self.child = token

    def __str__(self):
        return '(%s %s)' % (self.cat, self.child)

for line in sys.stdin:
    elements = line.strip().split()
    first_op = elements.pop(0)
    assert first_op == '[1/0]', "Line %s does not start with [1/0]" % line

    first_token = elements.pop(0)
    pt0 = PreTerminal(first_token)
    # Tree represents the current root of the tree
    tree = BinaryTree('X', pt0, rc=None, parent=None)
    # Frontier represents the lowest incomplete constituent
    frontier = tree

    while len(elements) > 1:
        op = elements.pop(0)
        op_f = int(op[1])
        op_j = int(op[3])
        token = elements.pop(0)
        pt = PreTerminal(token)

        # combine trees
        if op_f == 0 and op_j == 0:
            ## complete the right child with the current preterminal
            frontier.rc = pt
            ## And make the existing tree a left child of a new tree:
            parent = BinaryTree('X', lc=tree, rc=None, parent=None)
            tree.parent = parent
            ## And make the parent the base tree for next time:
            tree = parent
            ## The frontier is now the right child of the root tree:
            frontier = tree
        elif op_f == 1 and op_j == 1:
            ## extend the frontier:
            child = BinaryTree('X', pt, None, frontier)
            frontier.rc = child
            ## move the frontier to the child
            frontier = child
        else:
            raise Exception('This script does not yet handle the 1/0 or 0/1 cases!')

        # check for length issues
        if len(elements) == 1:
            raise Exception('This token list has an odd number of elements: %s\n' % line)
    
    print(tree)

