#include<iostream>
#include<vector>
#include<queue>
using namespace std;
struct Node{
    int key;
    Node *left,*right;
    Node(int k):key(k),left(0),right(0){}
};
vector<int> post1(100,0),post2(100,0);
int N;
void Create(Node *&root,int k){
    if(root==0){
        root = new Node(k);
        return ;
    }
    if(root->key > k){
        Create(root->left,k);
    }else
        Create(root->right,k);
}
int cnt = 0;
int flag = 1;
void postOrder(Node *root)
{
    if(root){
		postOrder(root->left);
        postOrder(root->right);
        if(flag==1)
            post1[cnt++] = root->key;
        else
            post2[cnt++] = root->key;
    }
}

void lvOrder(Node *root){
    queue<Node *> q;
    q.push(root);
    cnt = 0;
    while(!q.empty()){
        Node *t = q.front();q.pop();
        cout<<t->key;
        if(cnt < N-1) cout<<" ";
        cnt++;
        if(t->left) q.push(t->left);
        if(t->right) q.push(t->right);
    }
}

int main()
{
    cin>>N;
    int t;
    Node *root1 = 0;
    Node *root2 = 0;
    for(int i=0;i<N;i++){
        cin>>t;
        Create(root1,t);
    }
    for(int i=0;i<N;i++){
        cin>>t;
        Create(root2,t);
    }
    flag = 1;cnt = 0;
    postOrder(root1);
    flag = 2;cnt=0;
    postOrder(root2);
    bool flg = true;
    for(int i=0;i<N;i++){
        if(post1[i] != post2[i]){
            flg = false;
            break;
        }
    }
    cout<<(flg?"YES\n":"NO\n");
    for(int i=0;i<N;i++){
		cout<<post1[i];
		if(i<N-1) cout<<" ";
	}
    cout<<endl;
	cnt = 0;
    lvOrder(root1);
    cout<<endl;
    return 0;
}
