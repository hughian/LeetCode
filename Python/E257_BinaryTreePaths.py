# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pre(self, root, tmp):
        if not root: return
        if not root.left and not root.right:
            tmp += [root.val]
            self.ans += [tmp]
            tmp = []
        else:
            self.pre(root.left, tmp+[root.val])
            self.pre(root.right, tmp+[root.val])
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        ret = []
        self.ans = []
        self.pre(root, [])
        for L in self.ans:
            s = str(L[0])
            for t in L[1:]:
                s += '->' + str(t)
            ret += [s]
        return ret