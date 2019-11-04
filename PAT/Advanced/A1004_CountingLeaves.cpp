#include<iostream>
#include<vector>
#include<queue>
using namespace std;

int main()
{
    vector<vector<int> > tree;
    int n,m;
    cin>>n>>m;
    tree.resize(n+1);
    int id,k;
    for(int i=1;i<=m;i++){
        cin>>id>>k;
        tree[id].resize(k+1);
        tree[id][0] = k;
        for(int j=1;j<=k;j++)
            cin>>tree[id][j];
    }

    int tmp,sum = 0;
    vector<int> res;
    queue<int> q;
    q.push(1);
    q.push(-1);
    while(q.size()>1 ||(q.size()==1 && q.front() != -1)){
        tmp = q.front();q.pop();
        if(tmp == -1){
            res.push_back(sum);
            sum = 0;
            q.push(-1);
        }
        else{
            if(tree[tmp].size()<2)
                sum ++;
            else{
                for(int i=1;i<=tree[tmp][0];i++){
                    q.push(tree[tmp][i]);
                }
            }
        }
    }
    int i;
    for(i=0;i < (int)res.size();i++){
        cout<<res[i]<<" ";
    }
    cout<<sum;
    return 0;
}
