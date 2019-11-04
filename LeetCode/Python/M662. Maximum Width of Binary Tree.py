# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def pque(que):
    for q in que:
        if q is -1:
            print('-1',end=' ')
        elif q is None:
            print('null',end=' ')
        else:
            print('(%s, %s)' %(q[0].val, q[1]), end=' ')
    print('')


class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        if not root: return 0
        que = [(root, 1), None]
        m = 0
        level_first_index = 1
        last_index = 1
        while len(que) > 1:
            t = que.pop(0)
            if t is None:
                m = max(m, last_index - level_first_index + 1)
                # pque(que)
                que.append(None)
                level_first_index = que[0][1]
            else:
                rt, index = t
                if rt.left:
                    que.append((rt.left, 2 * index))
                if rt.right:
                    que.append((rt.right, 2 * index + 1))
                last_index = index
        return max(last_index - level_first_index + 1, m)
        
root = TreeNode(1)
root.left = TreeNode(3)
root.right = TreeNode(2)
root.left.left = TreeNode(5)
root.left.right = TreeNode(3)
root.right.right = TreeNode(9)
print(Solution().widthOfBinaryTree(root))
