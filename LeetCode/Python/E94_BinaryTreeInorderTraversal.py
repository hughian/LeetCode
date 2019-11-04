
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        
        ans = []
        
        def inorder_iterative(root):
            s = []
            ans = []
            p = root
            while p or len(s) != 0:
                while p:
                    s.append(p)
                    p = p.left
                if len(s) != 0:
                    p = s.pop()
                    ans += [p.val]
                    p = p.right
            return ans
            
        
        def inorder_recursive(root):
            nonlocal ans
            if root:
                inorder_recursive(root.left)
                ans += [root.val]
                inorder_recursive(root.right)
        inorder_recursive(root)
        return inorder_iterative(root)
        
root = TreeNode(0)
root.left = TreeNode(1)
root.left.left = TreeNode(3)
root.left.right = TreeNode(4)
root.right = TreeNode(2)
root.right.left = TreeNode(5)
root.right.right = TreeNode(6)



