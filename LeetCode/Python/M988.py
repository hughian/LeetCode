class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
MAP = lambda x: chr(x + ord('a'))

def get_str(cur_path):
    chars = [MAP(p.val) for p in cur_path[::-1]]
    return ''.join(chars)

class Solution:
    res = ""

    def dfs(cur_path):
        node = cur_path[-1]
        if node.left == node.right == None:
            tmp = get_str(cur_path)
            if self.res == "" or self.res > tmp:
                self.res = tmp
            return
        for x in [node.left, node.right]:
            if x:
                cur_path.append(x)
                dfs(cur_path)
                cur_path.pop()
    
    def preOrder(self, root, tmp):
        if root:
            if root.left == root.right == None:
                if self.res == "" or self.res > tmp:
                    self.res = tmp
            if root.left:
                self.preOrder(root.left, MAP(root.left.val) + tmp)
            if root.right:
                self.preOrder(root.right, MAP(root.right.val) + tmp)
    def get(self, root):
        if not root:
            return ""
        self.preOrder(root, MAP(root.val))
        return self.res

def smallestFromLeaf(root: TreeNode) -> str:
    stack = [root]
    mp = lambda x: chr(x + ord('a'))
    ttt = [mp(root.val)]
    p = root
    pre = None
    res = ""
    while len(stack) != 0:
        print(ttt)
        p  = stack.pop()
        _ = ttt.pop()
        if (p.left == p.right == None) or (pre == None and (pre == p.left or pre == p.right)):
            if p.left == p.right == None:
                tmp = ''.join(ttt[::-1])
                print(tmp)
                if res == "" or res > tmp:
                    res = tmp
        else:
            if p.right:
                stack.append(p.right)
                ttt.append(mp(p.right.val))
            if p.left:
                stack.append(p.left)
                ttt.append(mp(p.right.val))
    return res
 
root = TreeNode(25)
left = root.left = TreeNode(1)
right = root.right = TreeNode(3)
left.left = TreeNode(1)
left.right = TreeNode(3)
right.left = TreeNode(0)
right.right = TreeNode(2)

s = smallestFromLeaf(root)
print(s)

so = Solution()
print(so.get(root))
