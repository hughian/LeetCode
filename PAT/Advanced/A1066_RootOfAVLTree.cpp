#include<iostream>
using namespace std;
struct Node{
    int data;
    Node *left,*right;
    Node(int d):data(d),left(0),right(0){}
};

int getHeight(Node *root){
    if(root==0) return 0;
    else return max(getHeight(root->left),getHeight(root->right))+1;
}

void LL(Node *&root){
    Node *t = root->left;
    root ->left = t->right;
    t->right = root;
    root = t;
}

void RR(Node *&root){
    Node *t = root->right;
    root->right = t->left;
    t->left = root;
    root = t;
}

void LR(Node *&root){
    RR(root->left);
    LL(root);
}
void RL(Node *&root){
    LL(root->right);
    RR(root);
}

void insert(Node *&root,int d)
{
    if(root==0){
        root = new Node(d);return;
    }
    if(root->data <= d){
        insert(root->right,d);
        if(getHeight(root->right)-getHeight(root->left)==2){
            if(getHeight(root->right->right)-getHeight(root->right->left)==1)
                RR(root);
            else
                RL(root);
        }
    }else{
        insert(root->left,d);
        if(getHeight(root->left)-getHeight(root->right)==2){
            if(getHeight(root->left->left)-getHeight(root->left->right)==1)
                LL(root);
            else
                LR(root);
        }
    }
}

int main()
{
    int N,d;
    cin>>N;
    Node *root = 0;
    for(int i=0;i<N;i++){
        cin>>d;
        insert(root,d);
    }
    cout<<root->data;
    return 0;
}
