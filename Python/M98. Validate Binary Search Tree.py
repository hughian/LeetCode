class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
def isValidBST(root) -> bool:
    if not root: return True
    def pre(root):
        if not root: return True, float('inf'), -float('inf')
        if root:
            t = True
            minl, minr = float('inf'), float('inf')
            maxl, maxr = -float('inf'), -float('inf')
            if root.left:
                tt, minl, maxl = pre(root.left)
                t = t and tt and maxl < root.val
            if root.right:
                tt, minr, maxr = pre(root.right)
                t = t and tt and minr > root.val
            print(root.val, minl, maxl, minr, maxr)
            return t, min(minl, minr, root.val), max(maxl, maxr, root.val)
    return pre(root)[0]

def isValidBST(root):
    tmp = []
    def inorder(root, tmp):
        if root:
            inorder(root.left, tmp)
            tmp +=[root.val]
            inorder(root.right, tmp)
    inorder(root, tmp)
    print(tmp)
    return tmp == sorted(list(set(tmp)))

root = TreeNode(3)
root.right = TreeNode(30)
root.right.left = TreeNode(10)
root.right.left.right = TreeNode(15)
root.right.left.right.right = TreeNode(45)

r = isValidBST(root)
print(r)

T = TreeNode(5)
T.left = TreeNode(1)
T.right = TreeNode(4)
T.right.left = TreeNode(3)
T.right.right = TreeNode(6)

def ino(root):
    if root:
        ino(root.left)
        print(root.val, end=' ')
        ino(root.right)
ino(T)
