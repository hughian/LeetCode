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

//二叉树的层序遍历
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode *root) {
        if (root == nullptr)
            return vector<int>();
        queue < TreeNode * > que;
        vector<int> res;
        que.push(root);
        int lv = 0;
        while (!que.empty()) {
            int L = que.size();
            // cout<<L<<endl;
            for (int i = 0; i < L; i++) {
                TreeNode *t = que.front();
                que.pop();
                res.push_back(t->val);
                if (t->left)
                    que.push(t->left);
                if (t->right)
                    que.push(t->right);
            }
            lv += 1;
        }
        return res;
    }
};

// 题目二：多行打印
// 和上面的层序没区别
class Solution {
public:
    vector <vector<int>> Print(TreeNode *pRoot) {
        if (pRoot == nullptr)
            return vector < vector < int > > ();
        queue < TreeNode * > que;
        que.push(pRoot);
        TreeNode *t;
        vector <vector<int>> res;
        while (!que.empty()) {
            int L = que.size();
            vector<int> tmp;
            for (auto i = 0; i < L; i++) {
                t = que.front();
                que.pop();
                tmp.emplace_back(t->val);
                if (t->left)
                    que.push(t->left);
                if (t->right)
                    que.push(t->right);
            }
            res.emplace_back(tmp);
        }
        return res;
    }
};


// 题目三：之字形打印二叉树
// 用两个栈，代码写的比较丑
class Solution {
public:
    vector <vector<int>> Print(TreeNode *pRoot) {
        vector <vector<int>> res;
        if (pRoot == nullptr)
            return res;

        stack < TreeNode * > s1, s2;
        s1.push(pRoot);
        while (!s1.empty() || !s2.empty()) {
            vector<int> tmp;
            if (!s1.empty()) {
                int L = s1.size();
                for (int i = 0; i < L; i++) {
                    auto t = s1.top();
                    s1.pop();
                    tmp.push_back(t->val);
                    if (t->left)
                        s2.push(t->left);
                    if (t->right)
                        s2.push(t->right);
                }
                res.emplace_back(tmp);
            } else {
                int L = s2.size();
                for (int i = 0; i < L; i++) {
                    auto t = s2.top();
                    s2.pop();
                    tmp.push_back(t->val);
                    if (t->right)
                        s1.push(t->right);
                    if (t->left)
                        s1.push(t->left);
                }
                res.emplace_back(tmp);
            }
        }
        return res;
    }

};