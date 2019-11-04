#include<iostream>
#include<vector>
using namespace std;
vector<int> lv,in(40,0);
int N;
struct Node{
    int key;
    Node *left,*right;
    Node(int k):key(k),left(0),right(0){}
};
Node *build(vector<int> &lv,int low,int high){
    if(lv.size()==0) return 0;
    Node *root = new Node(lv[0]);
    int k =low;
    for(;k<=high;k++){
        if(in[k] == lv[0])
            break;
    }
    vector<int> leftLv,rightLv;
    for(int i=1;i<(int)lv.size();i++){
        bool isLeft = false;
        for(int j=low;j<k;j++){
            if(lv[i]==in[j]){
                isLeft = true;
                break;
            }
        }
        if(isLeft){
            leftLv.push_back(lv[i]);
        }else{
            rightLv.push_back(lv[i]);
        }
    }
    root->left = build(leftLv,low,k-1);
    root->right = build(rightLv,k+1,high);
    return root;
}

int cnt = 0;
void pre(Node *root){
    if(root){
        cout<<root->key;
        if(cnt<N-1) cout<<" ";
        cnt++;
        pre(root->left);
        pre(root->right);
    }
}
int ct = 0;
void post(Node *root){
    if(root){
        post(root->left);
        post(root->right);
        cout<<root->key;
        if(ct < N-1) cout<<" ";
        ct ++;
    }
}

int main()
{
    cin>>N;
    int t;
    for(int i=0;i<N;i++){ 
        cin>>t;
        lv.push_back(t);
    }
    for(int i=0;i<N;i++) cin>>in[i];
    Node *root = build(lv,0,N-1);
    pre(root);
    cout<<endl;
    post(root);
    cout<<endl;
    return 0;
}

