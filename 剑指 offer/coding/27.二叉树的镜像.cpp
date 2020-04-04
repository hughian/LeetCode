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
};*/

// 反转二叉树，后序遍历改一改
class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        if (pRoot){
            Mirror(pRoot->left);
            Mirror(pRoot->right);

            TreeNode *p = pRoot->left;
            pRoot->left = pRoot->right;
            pRoot->right = p;
        }
    }
};