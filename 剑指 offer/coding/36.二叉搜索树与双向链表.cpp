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


typedef pair<TreeNode *, TreeNode *> PTT;

// 先将左右子树链表化，然后链接根节点，最后返回合并后得头尾节点。
class Solution {
public:
    PTT helper(TreeNode *root) {
        if (root == nullptr) {
            return make_pair(nullptr, nullptr);
        }
        // 链表化左子树，返回头节点和尾节点
        PTT left = helper(root->left);
        root->left = left.second;
        if (left.second)
            left.second->right = root;

        // 链表化右子树，返回头节点和尾巴节点
        PTT right = helper(root->right);
        root->right = right.first;
        if (right.first)
            right.first->left = root;

        // 合并后的头尾节点
        if (left.first == nullptr && right.second == nullptr)
            return make_pair(root, root);
        else if (left.first == nullptr)
            return make_pair(root, right.second);
        else if (right.second == nullptr)
            return make_pair(left.first, root);
        else
            return make_pair(left.first, right.second);
    }

    TreeNode *Convert(TreeNode *pRootOfTree) {
        PTT p = helper(pRootOfTree);
        return p.first;
    }
};