from functools import cmp_to_key
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def verticalTraversal(self, root: TreeNode):
        r = dict()
        data = list()
        def pre(root, x, y):
            nonlocal data
            if root:
                data += [[x, y, root.val]]
                pre(root.left, x-1, y-1)
                pre(root.right, x+1, y-1)
        pre(root, 0, 0)

        def mycmp(a, b):
            if a[0] != b[0]: # x
                if a[0] < b[0]:
                    return -1
                else:
                    return 1
            elif a[1] != b[1]: # y
                if a[1] > b[1]:
                    return -1
                else:
                    return 1
            elif a[2] != b[2]: # val
                if a[2] < b[2]:
                    return -1
                else:
                    return 1
            else:
                return 0
        print(data)
        dd = sorted(data, key = cmp_to_key(mycmp))
        print(dd)
        ans = [ [dd[0][2]] ]
        for i in range(1, len(dd)):
            if dd[i][0] == dd[i-1][0]:
                ans[-1].append(dd[i][2])
            else:
                ans.append([dd[i][2]])
        print(ans)
        
        def preOrder(root, coord:tuple):
            if root:
                if coord not in r:
                    r[coord] = [root.val]
                else:
                    r[coord] += [root.val]
                preOrder(root.left, (coord[0]-1, coord[1]-1))
                preOrder(root.right, (coord[0]+1, coord[1]-1))

        preOrder(root, (0, 0))
        d = sorted(r.keys())
        print(d)
        ans = []
        last_x = None 
        for k in d:
            v = sorted(r[k])
            if last_x == k[0]:
                ans[-1] = v + ans[-1]
            else:
                ans.append(v)
            last_x = k[0]
        return ans
 
root = TreeNode(3)
left = root.left = TreeNode(9)
right = root.right = TreeNode(20)

right.left = TreeNode(15)
right.right = TreeNode(7)


so = Solution()
print(so.verticalTraversal(root))
