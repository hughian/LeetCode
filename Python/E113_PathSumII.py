# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
    def __repr__(self):
        return 'val:' + str(self.val) + ' ' + str(self.left) + ' ' + str(self.right)
        
class Solution(object):
    def pre(self, root, tmp):
        
        if not root.left and not root.right and sum(tmp)==self.sum:
            self.ans.append(tmp)
        else:
            if root.left: self.pre(root.left, tmp+[root.left.val])
            if root.right: self.pre(root.right, tmp+[root.right.val])
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        if root==None: return []
        self.sum = sum
        self.ans = []
        self.pre(root,[root.val])
        return self.ans
        
def main():
    root = TreeNode(5)
    root.left = TreeNode(4)
    root.right= TreeNode(8)
    root.left.left = TreeNode(11)
    root.right.left= TreeNode(13)
    root.right.right=TreeNode(4)
    root.left.left.left=TreeNode(7)
    root.left.left.right=TreeNode(2)
    root.right.right.left=TreeNode(5)
    root.right.right.right=TreeNode(1)
    
    s = Solution().pathSum(root,22)
    print(s)
    
if __name__=='__main__':
    main()