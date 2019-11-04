"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children
"""
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Node
        :rtype: int
        """
        def dfs(root, depth, maxdepth):
            if not root: return
            if depth > maxdepth[0]: maxdepth[0] = depth
            for c in root.children:
                dfs(c, depth+1, maxdepth)
        maxdepth = [0]
        dfs(root, 1, maxdepth)
        return maxdepth[0]