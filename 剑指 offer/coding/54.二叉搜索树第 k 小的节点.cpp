//
// Created by Hughian on 2020/4/4.
//

/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/

// 第 k 小/大都是一样的。中序遍历，找到第 k 个保留下结果就行了
class Solution {
public:
    void helper(TreeNode *&res, TreeNode *root, int k, int &i) {
        if (root != nullptr) {
            helper(res, root->left, k, i);
            if (i > k)
                return;
            i += 1;
            if (i == k) {
                res = root;
                return;
            }
            helper(res, root->right, k, i);
        }
    }

    TreeNode *KthNode(TreeNode *pRoot, int k) {
        TreeNode *res = nullptr;
        int i = 0;
        helper(res, pRoot, k, i);
        return res;
    }


};