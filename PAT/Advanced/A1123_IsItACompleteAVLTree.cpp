#include<iostream>
#include<queue>
using namespace std;

struct Node{
    int data;
    Node *left,*right;
    Node(int d):data(d),left(0),right(0){}
};

int N;

int getHeight(Node *root){
    if(root == 0) return 0;
    return (1 + max(getHeight(root->left),getHeight(root->right)));
}
Node* RR(Node *root){ //RR型，向左旋
    /*
       1             
        \              2
         2     ==>   /   \
          \         1     3
           3
    */
    Node *t = root->right;
    root->right = t -> left;
    t->left = root;
    return t;
}
Node* LL(Node *root){ //LL型，向右旋
    /*
        3
       /           2
      2    ==>   /   \
     /          1     3
    1
    */
    Node *t = root->left;
    root->left = t->right;
    t->right = root;
    return t;
}
Node* RL(Node *root){ //RL型，先向右，再向左
    /*
        1          1
         \          \             2
          3  ==>     2    ==>   /   \
         /            \        1     3
        2              3
    */
    root->right=LL(root->right); //右子树当做LL型向右旋
    root = RR(root);//第一步旋转后变为RR型，向左旋
    return root;
}
Node* LR(Node *root){ //RL型，先向左，再向右
    /*
        3           3
       /           /          2
      1     ==>   2    ==>  /   \
       \         /         1     3
        2       1
    */
    root->left = RR(root->left); //左子树当做RR型向左旋
    root = LL(root);//第一步旋转后变为LL型，向右旋
    return root;
}

void insert(Node *&root,int d){
    if(root==0){
        root = new Node(d);return;
    }
    if(d <= root->data){
        insert(root->left,d);
        if(getHeight(root->left)-getHeight(root->right) == 2) //左子树高于右子树
        {
            if(getHeight(root->left->left)-getHeight(root->left->right)==1){
                root = LL(root); //LL型，向右旋转
            }else{ //LR型，先左旋，再右旋
                root=LR(root);
            }
        }
    }else{
        insert(root->right,d);
        if(getHeight(root->right)-getHeight(root->left)==2){ //插入右子树，右高于左
            if(getHeight(root->right->right)-getHeight(root->right->left)==1){
                root = RR(root);//RR型,向左旋转
            }else{ //RL型，先右旋再左旋
                root=RL(root);
            }
        }
    }
}

bool lvorder(Node *root){
    bool first=false,flag = true;
    int cnt = 0;
    queue<Node *> q;
    q.push(root);
    while(!q.empty()){
        Node *t = q.front();q.pop();
        if(first){
            if(t->left != 0 || t->right != 0)
                flag = false;
        }else{
            if(t->right == 0)
                first = true;
            else if(t->left == 0)
                flag = false;
        }
        if(t->left != 0)
            q.push(t->left);
        if(t->right != 0)
            q.push(t->right);
        cout<<t->data<<((cnt<N-1)?" ":"");
        cnt++;
    }
    return flag;
}

int main()
{
    int d;
    cin>>N;
    Node *root=0;
    for(int i=0;i<N;i++){
        cin>>d;
        insert(root,d);
    }
    bool res = lvorder(root);
    cout<<endl;
    cout<<(res?"YES":"NO");
    return 0;
}
