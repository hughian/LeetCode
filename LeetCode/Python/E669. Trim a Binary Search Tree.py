# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def trimBST(self, root: TreeNode, L: int, R: int) -> TreeNode:
        
        def post(root):
            if not root:return None
            root.left = post(root.left)
            root.right = post(root.right)

            if root.val < L:
                root.left = None
                return root.right
            elif root.val > R:
                root.right = None
                return root.left
            else:
                return root
        return post(root)

def pre(root):
    if root:
        print(root.val, end=' ')
        pre(root.left)
        pre(root.right)
        
root = TreeNode(3)
root.left = TreeNode(0)
root.right = TreeNode(4)
root.left.right = TreeNode(2)
root.left.right.left = TreeNode(1)
s = Solution()
r = s.trimBST(root, 1, 3)
pre(r)