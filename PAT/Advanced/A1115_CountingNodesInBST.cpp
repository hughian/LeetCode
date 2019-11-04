#include<iostream>
using namespace std;

struct Node{
    int data;
    int lv;
    Node *left,*right;
    Node():data(0),lv(0),left(0),right(0){}
    Node(int d):data(d),lv(0),left(0),right(0){}
};
void insert(Node* root,int d){
    Node *p = root,*pre = root;
    Node *t = new Node(d);
    while(p){
        pre = p;
        if(d <= p->data)
            p = p->left;
        else
            p = p->right;
    }
    if(d <= pre->data)
        pre->left = t;
    else
        pre->right = t;
}
int maxl = -1;
void inorder(Node *tree,int lv){
    if(tree){
        inorder(tree->left,lv+1);
        tree->lv = lv;
        if(lv > maxl)
            maxl = lv;
        inorder(tree->right,lv+1);
    }
}
int n1 = 0;
int n2 = 0;
void preorder(Node *tree){
    if(tree){
        if(tree->lv == maxl)
            n1 ++;
        else if(tree->lv == maxl-1)
            n2 ++;
        preorder(tree->left);
        preorder(tree->right);
    }
}
int main()
{
    int n,d;
    cin>>n>>d;
    Node *root = new Node(d);
    for(int i=1;i<n;i++){
        cin>>d;
        insert(root,d);
    }
    inorder(root,1);
    preorder(root);
    cout<<n1<<" + "<<n2<<" = "<<n1+n2;
    return 0;
}
