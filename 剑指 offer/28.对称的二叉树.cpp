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

// 一个二叉树和他的镜像相同则是对称的，使用一个 helper func 来递归
class Solution {
public:
    bool helper(TreeNode* root1, TreeNode* root2){
        if(root1 == nullptr && root2 == nullptr)
            return true;
        if (root1 == nullptr || root2 == nullptr)
            return false;
        return (root1->val == root2->val) && helper(root1->left, root2->right);
    }
    bool isSymmetrical(TreeNode* pRoot)
    {
        return  helper(pRoot, pRoot);
    }

};