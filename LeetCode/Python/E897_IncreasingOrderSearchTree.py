# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorder(self, root):
        if not root: return
        self.inorder(root.left)
        self.inlist += [root.val]
        self.inorder(root.right)
    def increasingBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        self.inlist = []
        self.inorder(root)
        if len(self.inlist) == 0: return None
        r = TreeNode(self.inlist[0])
        t = r
        for i in self.inlist[1:]:
            t.right=TreeNode(i)
            t = t.right
        return r