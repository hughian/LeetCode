#include<iostream>
#include<vector>
using namespace std;

vector< vector<int> > cont;
vector<int> weight;
vector<int> order;
vector<int> Rank;
int Np,Ng;
int main()
{
    cin>>Np>>Ng;
    weight.resize(Np);
    order.resize(Np);
    Rank.resize(Np,-1);
    for(int i=0;i<Np;i++)
        cin>>weight[i];
    for(int i=0;i<Np;i++)
        cin>>order[i];
    vector<int> tmp;
    cont.push_back(order);
    while(order.size()>1){
        int i=0;
        tmp.clear();
        int len = order.size();
        while(i<(int)order.size()){
            int max = 0,maxi = 0;
            for(int j=i;j<i+Ng && j<len;j++){
                if(weight[order[j]]>max){
                    max = weight[order[j]];
                    maxi = order[j];
                }
            }
            tmp.push_back(maxi);
            i += Ng;
        }
        order = tmp;
        cont.push_back(order);
    }
    int cnt=1,n;
    for(int i=cont.size()-1;i>=0;i--){
        n = 0;
        for(unsigned j=0;j<cont[i].size();j++){
            int idx = cont[i][j];
            if(Rank[idx]==-1){
                Rank[idx] = cnt;
                n++;
            }
        }
        cnt+=n;
    }
    for(int i=0;i<Np;i++){
        cout<<Rank[i];
        if(i<Np-1) cout<<" ";
    }
    return 0;
}
