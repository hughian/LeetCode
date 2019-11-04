#include<iostream>
#include<vector>
#include<string>
using namespace std;
int N;
vector<int> lc(30,-1);
vector<int> rc(30,-1);
vector<int> parent(30,-1);
vector<string> strd(30,"");
string ans;
bool flag;
void inorder(int root){
    if(root!=-1){
        bool flag = (rc[root] != -1) ;
        if(flag){
            //cout<<"(";
            ans+="(";
        }
        inorder(lc[root]);
        //cout<<strd[root];
        ans+=strd[root];
        inorder(rc[root]);
        if(flag){
            //cout<<")";
            ans+=")";
        }
    }
}

int main(){
    cin>>N;
    string d;
    int l,r;
    for(int i=1;i<=N;i++){
        cin>>d>>l>>r;
        strd[i] = d;
        lc[i] = l;
        rc[i] = r;
        if(l!=-1) parent[l] = i;
        if(r!=-1) parent[r] = i;
    }
    int root=1;
    for(;root<=N;root++){
        if(parent[root]==-1)
            break;
    }
	if(lc[root] == -1 && rc[root]==-1){
		cout<<strd[root];
	}else{
		inorder(root);
		cout<<ans.substr(1,ans.length()-2);
    }
	return 0;
}
