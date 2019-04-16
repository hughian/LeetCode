/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    void postOrder(TreeNode* root, int val, bool &res){
        if(root){
            postOrder(root->left,val,res);
            postOrder(root->right,val,res);
            res &= (root->val == val);
        }
    }
    bool isUnivalTree(TreeNode* root) {
        bool res = true;
        postOrder(root,root->val,res);
        return res;
    }
};