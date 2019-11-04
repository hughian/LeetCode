#include<iostream>
#include<vector>
#include<string>
#include<cstdlib>
#include<queue>
using namespace std;

int mystoi(string str){
    if(str[0]=='-')
        return -1;
    else
        return stoi(str);
}

vector<int> parent(31,-1);
vector<int> lchild(31,-1);
vector<int> rchild(31,-1);
bool flag;
int isCompleteTree(int root){ //判断是否是完全二叉树的方法，层序遍历时，只有左子树的非叶子结点和叶结点，其后的所有结点都是叶节点
    queue<int> q;
    q.push(root);
    bool isNul = false;
    int t;
    while(!q.empty()){
        t = q.front();q.pop();
        if(isNul){
            if(lchild[t]!=-1 || rchild[t]!=-1)
                flag = false;
        }else{
            if(lchild[t]!=-1 && rchild[t]!=-1){
                q.push(lchild[t]);
                q.push(rchild[t]);
            }else if(lchild[t]!=-1 && rchild[t]==-1){
                q.push(lchild[t]);
                isNul = true;
            }else if(lchild[t]==-1 && rchild[t] != -1){
                flag = false;
            }else{
                isNul = true;
            }
        }
    }
    return t;
}

int main()
{
    int n;
    cin>>n;
    string l,r;
    int lf,rt;
    for(int i=0;i<n;i++){
        cin>>l>>r;
        lf = mystoi(l);
        rt = mystoi(r);
        lchild[i] = lf;
        rchild[i] = rt;
        if(lf != -1)
            parent[lf] = i;
        if(rt != -1)
            parent[rt] = i;
    }
    int root;
    for(int i=0;i<n;i++){
        if(parent[i]==-1){
            root = i;break;
        }
    }
    flag = true;
    int last = isCompleteTree(root);
    if(flag)
        cout<<"YES "<<last;
    else
        cout<<"NO "<<root;
    return 0;
}
