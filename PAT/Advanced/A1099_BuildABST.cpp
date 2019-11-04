#include<iostream>
#include<vector>
#include<algorithm>
#include<queue>
using namespace std;
vector<int> lc(101,-1);
vector<int> rc(101,-1);
vector<int> pr(101,-1);
vector<int> dt(101,-1);
vector<int> tmp(101,-1);
int N;
int cnt = 0;
void inOrder(int root){
    if(root!=-1){
        inOrder(lc[root]);
        dt[root] = tmp[cnt++];
        inOrder(rc[root]);
    }
}
void lvOrder(int root){
    queue<int> q;
    q.push(root);
    cnt = 0;
    while(!q.empty()){
        int t = q.front();q.pop();
        cout<<dt[t];
        if(cnt<N-1)
            cout<<" ";
		cnt++;
        if(lc[t]!=-1) q.push(lc[t]);
        if(rc[t]!=-1) q.push(rc[t]);
    }
}

int main()
{
    cin>>N;
    int li,ri;
    for(int i=0;i<N;i++){
        cin>>li>>ri;
        lc[i] = li;
        rc[i] = ri;
        if(li!=-1) pr[li] = i;
        if(ri!=-1) pr[ri] = i;
    }
    for(int i=0;i<N;i++)
        cin>>tmp[i];
    sort(tmp.begin(),tmp.begin()+N);
    inOrder(0);
    lvOrder(0);
    return 0;
}
