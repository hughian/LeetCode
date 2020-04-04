//
// Created by Hughian on 2020/4/4.
//


struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};

// 检查 树2 是不是 树1 的子结构，递归就完了
// 注意检查空指针，这是最难的地方
class Solution {
public:
    bool check(TreeNode* p1, TreeNode* root2){
        if (root2 == nullptr)
            return true;
        if (p1 == nullptr)
            return false;

        if (p1->val != root2->val){
            return false;
        }

        return check(p1->left, root2->left) && check(p1->right, root2->right);
    }
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if (pRoot1 && pRoot2){
            if (pRoot1->val == pRoot2->val){
                return check(pRoot1, pRoot2)|| (HasSubtree(pRoot1->left, pRoot2) || HasSubtree(pRoot1->right, pRoot2));
            }else{
                return (HasSubtree(pRoot1->left, pRoot2) || HasSubtree(pRoot1->right, pRoot2));
            }
        }
        return false;
    }
};