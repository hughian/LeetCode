# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# 56 ms
class Solution:
    def postOrder(self, root, val, res):
        if root:
            self.postOrder(root.left, val, res)
            self.postOrder(root.right,val, res)
            res[0] &= (root.val == val)
    def isUnivalTree(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        res = [True]
        self.postOrder(root, root.val, res)
        return res[0]
        