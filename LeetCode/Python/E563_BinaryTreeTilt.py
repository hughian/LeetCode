# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def sumNode(self, root):
        if not root: return 0,0
        left,tilt = self.sumNode(root.left)
        self.ans += tilt
        right,tilt = self.sumNode(root.right)
        self.ans += tilt
        tilt = abs(left - right)
        return left+right+root.val, tilt
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.ans = 0
        if not root: return 0
        s,t = self.sumNode(root)
        self.ans += t
        return self.ans