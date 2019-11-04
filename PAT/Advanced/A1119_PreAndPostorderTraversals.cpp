#include<iostream>
#include<vector>

using namespace std;
struct Node{
    int data;
    Node *left,*right;
    Node(int d):data(d),left(0),right(0){}
};

vector<int> pre;
vector<int> post;
int n;
bool flag = true;
Node *buildtree(int preL,int preH,int postL,int postH)
{
    if(preL>preH || postL>postH)
        return NULL;
    Node *tree = new Node(pre[preL]);
    if(preL == preH)
        return tree;

    //ͨ�����Һ������������һ������ǰһ���������е�λ�ã�
    //��ȷ���Ƿ���Ի������Һ��ӣ�
    //  ������ܣ��ͽ��仮��Ϊ�Һ��ӣ������ӣ���
    //�ݹ齨��
    int i=preL+1;
    while(i <= preH){
        if(pre[i]==post[postH-1])
            break;
        i++;
    }
    if(i-preL>1){
        tree->left = buildtree(preL+1,i-1,postL,postL+(i-1)-(preL+1));
        tree->right = buildtree(i,preH,postL+(i-1)-(preL+1)+1,postH-1);
    }else{
        flag = false;
        tree->right = buildtree(i,preH,postL+(i-1)-(preL+1)+1,postH-1);
    }
    return tree;
}
int cnt = 0;
void inorder(Node *root){
    if(root){
        inorder(root->left);
        cout<<root->data;
        if(cnt < n-1) //��ʽ��������һ����������󲻴��ո�
            cout<<" ";
		cnt++;
        inorder(root->right);
    }
}

int main()
{
    cin>>n;
    pre.resize(n);
    post.resize(n);
    for(int i=0;i<n;i++)
        cin>>pre[i];
    for(int i=0;i<n;i++)
        cin>>post[i];
    Node *root = buildtree(0,n-1,0,n-1);
    if(flag)
		cout<<"Yes"<<endl;
	else
		cout<<"No"<<endl;
    inorder(root);
    cout<<endl;
    return 0; 
}
