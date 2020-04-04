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

// dfs 保存所有的路径，然后对路径按长度进行排序
class Solution {
    int sum = 0;
    void dfs(TreeNode *root, vector<vector<int> > &res, vector<int> path, int s){
        if(root){
            if(s + root->val == sum && root->left==nullptr && root->right==nullptr){
                path.push_back(root->val);
                res.emplace_back(path);
            }else{
                path.push_back(root->val);
                dfs(root->left, res, path, s+root->val);
                dfs(root->right, res, path, s+root->val);
                path.pop_back();
            }
        }
    }
    static int cmp(vector<int>&a, vector<int>&b){
        return a.size() > b.size(); // 注意 cmp 函数的写法
    }
public:
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
        vector<vector<int> > s;
        sum = expectNumber;
        dfs(root, s, vector<int>(), 0);
        sort(s.begin(), s.end(), cmp);
        return s;
    }
};