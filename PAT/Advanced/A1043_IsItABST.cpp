#include<iostream>
#include<vector>
using namespace std;
vector<int> vn(1010);
vector<int> pre(1010);
vector<int> rpre(1010);
int N,t;
struct Node{
    int key;
    Node *left,*right;
    Node(int k):key(k),left(0),right(0){}
};

void insert(Node *root,int k){
    Node *p = root,*pre;
    while(p){
        pre = p;
        if(p->key <= k)
            p = p->right;
        else
            p = p->left;
    }
    if(pre->key <= k)
        pre->right = new Node(k);
    else
        pre->left = new Node(k);
}
int cnt = 0,rcnt=0,pcnt = 0;
void preOrder(Node *root){
    if(root){
        pre[cnt++] = root->key;
        preOrder(root->left);
        preOrder(root->right);
    }
}
void RpreOrder(Node *root){
    if(root){
        rpre[rcnt++] = root->key; 
        RpreOrder(root->right);
        RpreOrder(root->left);
    }
}

void postOrder(Node *root){
    if(root){
        if(t==-1){
            postOrder(root->left);
            postOrder(root->right);
        }else if(t==1){
            postOrder(root->right);
            postOrder(root->left);
        }
        cout<<root->key;
        if(pcnt < N-1) cout<<" ";
        pcnt++;
    }
}

int cmp(){
    bool pref = true,rpref = true;
    for(int i=0;i<N;i++)
        if(vn[i] != pre[i]){
            pref = false;break;
        }
    for(int i=0;i<N;i++)
        if(vn[i] != rpre[i]){
            rpref = false;break;
        }
    if(pref)
        return -1;
    else if(rpref)
        return 1;
    else
        return 0;
}

int main()
{
    cin>>N;
    for(int i=0;i<N;i++) cin>>vn[i];
    Node *root = new Node(vn[0]);
    for(int i=1;i<N;i++)
        insert(root,vn[i]);
    preOrder(root);
    RpreOrder(root);
    t = cmp();
    if(t){
        cout<<"YES\n";
        postOrder(root);
    }else
        cout<<"NO\n";
    return 0;
}
