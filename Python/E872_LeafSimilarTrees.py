# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def leafSimilar(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: bool
        """
        def preOrder(root, L):
            if not root: return
            if not root.left and not root.right:
                L += [root.val]
            preOrder(root.left, L)
            preOrder(root.right, L)
            
        p, q = [], []
        preOrder(root1, p)
        preOrder(root2, q)
        return p==q

def stringToTreeNode(input):
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(',')]
    root = TreeNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = TreeNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = TreeNode(rightNumber)
            nodeQueue.append(node.right)
    return root

def main():
    import sys
    def readlines():
        for line in sys.stdin:
            yield line.strip('\n')

    lines = readlines()
    while True:
        try:
            line = next(lines)
            root1 = stringToTreeNode(line);
            line = next(lines)
            root2 = stringToTreeNode(line);
            
            ret = Solution().leafSimilar(root1, root2)

            out = (ret);
            print(out)
        except StopIteration:
            break

if __name__ == '__main__':
    main()
    