class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        def pre(root):
            if not root: return None, None
            if root.left == root.right == None: return root, root
            elif root.left is None:
                right_head, right_tail = pre(root.right)
                return root, right_tail
            elif root.right is None:
                left_head, left_tail = pre(root.left)
                root.right = left_head
                root.left = None
                return root, left_tail
            else:
                left_head, left_tail = pre(root.left)
                right_head, right_tail = pre(root.right)
                left_tail.right = right_head
                root.left = None
                root.right = left_head
                return root, right_tail
        pre(root)

def pre(root):
    if root:
        print(root.val, end=' ')
        pre(root.left)
        pre(root.right)
    else:
        print('null', end=' ')

def post(root):
    if root:
        post(root.left)
        post(root.right)
        print(root.val, end=' ')

def _post(root):
    stack = []
    res = []
    p = root
    last = root
    while stack or p:
        while p:
            stack.append(p)
            p = p.left
        if stack:
            p = stack[-1]
            if p.right == None or last == p.right:
                p = stack.pop()
                res += [p.val]
                last = p
                p = None
            else:
                p = p.right
    return res


    

root = TreeNode(1)
root.left = TreeNode(2)
root.left.left = TreeNode(3)
root.left.right = TreeNode(4)
root.right = TreeNode(5)
root.right.right = TreeNode(6)
pre(root)
print('')
s = Solution()
s.flatten(root)
pre(root)

root = TreeNode(1)
root.left = TreeNode(2)
root.left.left = TreeNode(3)
root.left.right = TreeNode(4)
root.right = TreeNode(5)
root.right.right = TreeNode(6)

print('')
post(root)
print('')
print(_post(root))
