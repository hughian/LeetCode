class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def minCameraCover(self, root: 'TreeNode') -> 'int':
        
        cam = 0
        def pre(root):
            nonlocal cam
            if not root: return 0
            left = pre(root.left)
            right = pre(root.right)
            if left < 0 or right < 0:
                cam += 1
                return 1
            return max(left,right) - 1
        r = pre(root)
        if r < 0:
            cam += 1
        return cam
        
root = TreeNode(0)
root.left = TreeNode(0)
root.left.left = TreeNode(0)
root.left.right = TreeNode(0)

tree = TreeNode(0)

s = Solution()
r = s.minCameraCover(root)
print(r)

r = s.minCameraCover(tree)
print(r)

