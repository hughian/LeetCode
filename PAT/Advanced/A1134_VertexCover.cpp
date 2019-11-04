#include<iostream>
#include<set>
#include<vector>

using namespace std;

int main()
{
    int n,m;
    cin>>n>>m;
    vector< vector<int> > edge(m,vector<int>(2,0));
    for(int i=0;i<m;i++){
        cin>>edge[i][0]>>edge[i][1];
    }
    int k;
    cin>>k;
    int nv,tmp;
    set<int> iset;
    for(int i=0;i<k;i++){
        cin>>nv;
        iset.clear();
        for(int j=0;j<nv;j++){
            cin>>tmp;
            iset.insert(tmp);
        }
        int flag  =1;
        for(int j=0;j<m;j++){
            if(iset.count(edge[j][0])==0 && iset.count(edge[j][1])==0){
                flag = 0;
                break;
            }
        }
        if(flag)
            cout<<"Yes"<<endl;
        else
            cout<<"No"<<endl;
    }
}

