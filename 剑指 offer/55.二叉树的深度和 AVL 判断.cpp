//
// Created by Hughian on 2020/4/4.
//

// 后序遍历就完了
class Solution {
public:
    int TreeDepth(TreeNode* pRoot){
        if (pRoot == nullptr) return 0;

        int left = TreeDepth(pRoot->left);
        int right = TreeDepth(pRoot->right);
        return ::max(left, right) + 1;
    }
};


// AVL 的判断
class Solution {
    bool flag = true;
public:
    int height(TreeNode *root){
        if (root == nullptr)
            return 0;
        int left = height(root->left);
        int right = height(root->right);
        int diff = abs(left - right);
        if (diff > 1)
            flag = false;
        return ::max(left, right) + 1;
    }
    bool IsBalanced_Solution(TreeNode* pRoot) {
        if (pRoot == nullptr)
            return true;
        height(pRoot);
        return flag;
    }
};