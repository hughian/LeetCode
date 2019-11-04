#include<iostream>
#include<vector>
#include<string>
#include<stack>
using namespace std;

struct Node{
    int data;
    Node *left,*right;
    Node(int d):data(d),left(0),right(0){}
    Node():data(0),left(0),right(0){}
};

vector<int> pre;
vector<int> in;
int N;
Node* build(int preL,int preR,int inL,int inR)
{
    if(preL>preR)
        return NULL;
    Node *tree = new Node(pre[preL]);
    int i=inL;
    for(;i<inR;i++){
        if(in[i]==pre[preL]) break;
    }
    tree->left = build(preL+1,preL+i-inL,inL,i-1);
    tree->right= build(preL+i-inL+1,preR,i+1,inR);
    return tree;
}
int cnt = 0;
void postOrder(Node *root){
    if(root){
        postOrder(root->left);
        postOrder(root->right);
        cout<<root->data;
        if(cnt<N-1) cout<<" ";
        cnt++;
    }
}
int main()
{
    cin>>N;
    string op;
    int d;
    stack<int> s;
    for(int i=0;i<2*N;i++){
        cin>>op;
        if(op=="Push"){
            cin>>d;
            s.push(d);
            pre.push_back(d);
        }else{
            d = s.top();s.pop();
            in.push_back(d);
        }
    }
    Node* root = build(0,N-1,0,N-1);
    postOrder(root);
    return 0;
}
