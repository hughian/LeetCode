#include<iostream>
#include<vector>
#include<string>
#include<queue>
using namespace std;

vector<int> lc(20,-1);
vector<int> rc(20,-1);
vector<int> pr(20,-1);
int N;

int mystoi(string str)
{
    if(str[0]=='-')
        return -1;
    return stoi(str);
}
void Invert(int root){
    if(root != -1){
        Invert(lc[root]);
        Invert(rc[root]);
        int tmp = lc[root];
        lc[root] = rc[root];
        rc[root] = tmp;
    }
}

void lvOrder(int root){
    if(root==-1) return ;
    queue<int> q;
    q.push(root);
    int cnt = 0;
    while(!q.empty()){
        int t = q.front();q.pop();
        cout<<t;
        if(cnt<N-1)
            cout<<" ";
        cnt++;
        if(lc[t] != -1) q.push(lc[t]);
        if(rc[t] != -1) q.push(rc[t]);
    }
}
int Cnt = 0;
void inOrder(int root){
    if(root!=-1){
        inOrder(lc[root]);
        cout<<root;
        if(Cnt<N-1)
            cout<<" ";
        Cnt++;
        inOrder(rc[root]);
    }
}

void print(vector<int> &v){
    for(int i=0;i<N;i++)
        cout<<v[i]<<"\t";
    cout<<endl;
}

int main(){
    cin>>N;
    string ls,rs;
    for(int i=0;i<N;i++){
        cin>>ls>>rs;
        int li = mystoi(ls),ri = mystoi(rs);
        rc[i] = li;
        lc[i] = ri; //直接交换每个根的子树
        if(li!=-1) pr[li] = i;
        if(ri!=-1) pr[ri] = i;
    }
    int root = 0;
    for(;root<N;root++)
        if(pr[root]==-1)
            break;
    //Invert(root);
    lvOrder(root);
    cout<<endl;
    inOrder(root);
    return 0;
}
