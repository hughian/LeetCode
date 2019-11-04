#include<iostream>
using namespace std;
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution {
public:
    int count;
    int countNodes(TreeNode* root) {
        count=0;
		int depth=0;
		int ctmp;
		int i;
		TreeNode* p=root;
		while(p){
			depth++;
			p=p->left;
		}
        if(depth=0)
			return 0;
		else if(depth<=6){
			DFS(root);
			return count;
		}
		else{
			i=1;
			ctmp=1;
			while(i<depth){
				count += ctmp;
				ctmp *= 2;
				i++;
			}
			
			
		}
		
        return count;
    }
    void DFS(TreeNode *root){
        if(root){
            count++;
            DFS(root->left);
            DFS(root->right);
        }
    }
};
