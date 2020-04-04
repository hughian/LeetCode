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
class Solution {
    int index = -1;
public:
    string Serialize(TreeNode *root) {
        if (root == nullptr)
            return "#";
        return to_string(root->val) + "," + Serialize(root->left) + "," + helper(root->right);
    }

    TreeNode *Parse(vector <string> &vec) {
        index++;
        if (index >= vec.size())
            return nullptr;
        TreeNode *root = nullptr;
        if (vec[index] != "#") {

            root = new TreeNode(stoi(vec[index]));
            root->left = Parse(vec);
            root->right = Parse(vec);
        }
        return root;
    }

    TreeNode *Deserialize(char *str) {
        // C++ 还要自己分字符串
        string s(str);
        string sep(",");
        string token;
        size_t pos;
        vector <string> vec;
        while ((pos = s.find(sep)) != string::npos) {
            token = s.substr(0, pos);
            vec.emplace_back(token);
            s.erase(0, pos + 1);
        }
        // 建树
        return Parse(vec);
    }
};