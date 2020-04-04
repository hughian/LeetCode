//
// Created by Hughian on 2020/4/4.
//

// Definition for binary tree
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

// 递归建树，每个子树的前序序列第一个值是子树的根节点，然后在中序中找到相应位置，划分子树就可以了。
class Solution {
public:
    TreeNode* helper(vector<int> &pre, vector<int> &vin, int pi, int pj, int vi, int vj){
        if(pi > pj){
            return nullptr;
        }
        TreeNode* root = new TreeNode(pre[pi]);
        int k = vi;
        for(;k<vj;k++){
            if(vin[k] == pre[pi])
                break;
        }
        root->left = helper(pre, vin, pi+1, pi+k-vi, vi, k-1);
        root->right = helper(pre, vin, pi+k-vi+1, pj, k+1, vj);
        return root;
    }

    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        return helper(pre, vin, 0, pre.size()-1, 0, vin.size()-1);
    }
};