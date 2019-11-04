#include<iostream>
#include<vector>
using namespace std;
struct Node{
	int key;
	bool isRed;
	Node *left,*right;
	Node(int k,bool red):key(k),isRed(red),left(0),right(0){}
};

void build(Node *&root,int k,bool red){
	if(root==0){
		root = new Node(k,red);
		return;
	}
	if(root->key > k)
		build(root->left,k,red);
	else
		build(root->right,k,red);
}
void freeM(Node *root){
	if(root){
		freeM(root->left);
		freeM(root->right);
		delete root;
		root = 0;
	}
}
bool first = true,flag = true;
int num;
void preCheck(Node *root,bool isRedparent,int blackNum){
	if(root){
		if(isRedparent && root->isRed){
			flag = false;
		}
		int t = root->isRed?0:1;
		preCheck(root->left,root->isRed,blackNum+t);
		preCheck(root->right,root->isRed,blackNum+t);
	}else{
		if(first){
			num = blackNum;
			first = false;
		}else if(num != blackNum){
				flag = false;
		}
		
	}
}

void check(Node *root){
	if(root->isRed) flag = false;
	else	preCheck(root,false,0);
}

int main()
{
	int K,N;
	cin>>K;
	int t;
	bool isRed;
	Node *root = 0;
	for(int i=0;i<K;i++){
		cin>>N;
		root = 0;
		first = true;
		flag = true;
		for(int j=0;j<N;j++){
			cin>>t;
			isRed = false;
			if(t<0){
				isRed=true;
				t = -t;
			}
			build(root,t,isRed);
		}
		check(root);
		cout<<(flag?"Yes\n":"No\n");
		freeM(root);
	}
	
}