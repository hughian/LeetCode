#include<iostream>
#include<vector>
#include<queue>
using namespace std;
struct Node{
    int key;
    Node *left,*right;
    Node(int k):key(k),left(0),right(0){}
};
vector<int> post(40,0);
vector<int> in(40,0);
int N;

Node *build(int postL,int postR,int inL,int inR)
{
    if(postL>postR) return 0;
    Node *root = new Node(post[postR]);
    int k = inL;
    for(;k<inR;k++){
        if(in[k] == post[postR])
            break;
    }
    int len = k - inL;
    root->left = build(postL,postL+len-1,inL,k-1);
    root->right = build(postL+len,postR-1,k+1,inR);
    return root;
}
void lvOrder(Node *root){
    queue<Node *> q;
    q.push(root);
    int cnt = 1;
    while(!q.empty()){
        Node *t = q.front();q.pop();
        if(t->left) q.push(t->left);
        if(t->right) q.push(t->right);
        cout<<t->key;
        if(cnt<N) cout<<" ";
        cnt++;
    }
}

int main()
{
    cin>>N;
    for(int i=0;i<N;i++) cin>>post[i];
    for(int i=0;i<N;i++) cin>>in[i];
    Node *root=build(0,N-1,0,N-1);
    lvOrder(root);
    return 0;
}
