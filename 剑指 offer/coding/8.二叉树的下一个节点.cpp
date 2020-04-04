//
// Created by Hughian on 2020/4/4.
//


/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next; // 夫节点
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {

    }
};
*/

// 中序遍历的下一个节点并返回
class Solution {
public:
    TreeLinkNode *GetNext(TreeLinkNode *pNode) {
        // edge case
        if (pNode == nullptr)
            return nullptr;

        if (pNode->right == nullptr) {
            TreeLinkNode *p = pNode->next; // 父节点
            while (p && pNode == p->right) { // 只要是当前节点是其父节点的右节点，就向上
                pNode = p;
                p = p->next;
            }
            return pNode->next; // 直到是从左子树返回的
        }
        // 当前节点右子树不为空，那就直接右子树的最左下一个节点
        TreeLinkNode *p = pNode->right, *prev = nullptr;
        while (p) {
            prev = p;
            p = p->left;
        }
        return prev;
    }
};