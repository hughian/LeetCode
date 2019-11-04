#include<iostream>
#include<vector>
#include<queue>
using namespace std;
vector<int> in(40,0);
vector<int> post(40,0);
int N;
struct Node{
    int data;
    Node *left,*right;
    Node(int d):data(d),left(0),right(0){}
};

Node* buildTree(int inL,int inR,int postL,int postR)
{
    if(inL > inR || postL>postR)
        return 0;
    Node *tree = new Node(post[postR]);
    int i=inL;
    for(;i<=inR;i++){
        if(in[i] == post[postR]) break;
    }
    tree->left = buildTree(inL,i-1,postL,postL+i-inL-1);
    tree->right= buildTree(i+1,inR,postL+i-inL,postR-1);
    return tree;
}

void zigzag(Node *root){
    queue<Node *> q;
    q.push(root);
    q.push(0);
    bool left = false;
    int cnt = 1;
    vector<int> ans;
    vector<int> tmp;
    while(q.size()>1){
        Node *t = q.front();q.pop();
        if(t){
            tmp.push_back(t->data);
            if(t->left) q.push(t->left);
            if(t->right) q.push(t->right);
        }else{
            if(left){
                for(unsigned i=0;i<tmp.size();i++){
                    ans.push_back(tmp[i]);
                }
            }else{
                for(int i=tmp.size()-1;i>=0;i--){
                    ans.push_back(tmp[i]);
                }
            }
            tmp.clear();
            q.push(0);
            left = !left;
        }
    }
    if(left){
        for(unsigned i=0;i<tmp.size();i++)
            ans.push_back(tmp[i]);
    }else{
        for(int i=tmp.size()-1;i>=0;i--)
            ans.push_back(tmp[i]);
    }
    for(unsigned i=0;i<ans.size();i++){
        cout<<ans[i];
        if(cnt<(int)ans.size()) cout<<" ";
        cnt++;
    }
}

void inoder(Node *root){
    if(root){
        inoder(root->left);
        cout<<root->data<<" ";
        inoder(root->right);
    }
}
void postorder(Node *root){
    if(root){
        postorder(root->left);
        postorder(root->right);
        cout<<root->data<<" ";
    }
}
int main()
{
    cin>>N;
    for(int i=0;i<N;i++)
        cin>>in[i];
    for(int i=0;i<N;i++)
        cin>>post[i];
    Node *root = buildTree(0,N-1,0,N-1);
    zigzag(root);
    /*cout<<endl;
    inoder(root);
    cout<<endl;
    postorder(root);
    */
    return 0;
}
