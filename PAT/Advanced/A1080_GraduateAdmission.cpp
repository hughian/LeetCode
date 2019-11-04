#include<iostream>
#include<vector>
#include<algorithm>
#include<cstdio>
using namespace std;
int printf(const char *,...);
struct App{
    int id;
    int Ge,Gi;
    float G;
    vector<int> choices;
    int rank;
    App(int i,int ge,int gi,vector<int> c)
        :id(i),Ge(ge),Gi(gi),choices(c),rank(0){G = (Ge + Gi) / 2.0;}
    bool operator < (const App &a){
        if(this->G == a.G){
            return this->Ge > a.Ge;
        }
        return this->G > a.G;
    }
};
vector<App> va;
int N,M,K;
vector<int> quota;
vector< vector<int> > ans;
void print(vector<int> &v){
    unsigned len = v.size();
    for(unsigned i=0;i<len;i++)
        cout<<v[i]<<(i<len-1?" ":"");
    cout<<endl;
}
void print(vector<App> &v)
{
    for(unsigned i=0;i<v.size();i++){
        printf("%2d\t%d\t%d\t%.2f",i,v[i].Ge,v[i].Gi,v[i].G);
        for(unsigned j=0;j<v[i].choices.size();j++){
            printf("\t%d",v[i].choices[j]);
        }
        printf("\t%d\n",v[i].rank);
    }
}

int main()
{
    cin>>N>>M>>K;
    quota.resize(M);
    ans.resize(M);
    for(int i=0;i<M;i++)
        cin>>quota[i];
    vector<int> tpv(K);
    int ge,gi;
    for(int i=0;i<N;i++){
        cin>>ge>>gi;
        for(int j=0;j<K;j++)
            cin>>tpv[j];
        va.push_back(App(i,ge,gi,tpv));
    }
    sort(va.begin(),va.end());
    
    va[0].rank = 1;
    for(unsigned i=1;i<va.size();i++){
        if(va[i].Ge == va[i-1].Ge && va[i].G == va[i-1].G)
            va[i].rank = va[i-1].rank;
        else
            va[i].rank = i + 1;
    }
    //print(quota);
    //print(va);
    quota[va[0].choices[0]] --;
    ans[va[0].choices[0]].push_back(va[0].id);
    int last_rank = 1;
    int last_c = va[0].choices[0];
    for(unsigned i=1;i<va.size();i++){
        for(int j =0;j<K;j++){
			int cur_c = va[i].choices[j];
            if(quota[cur_c]>0 || (cur_c==last_c && va[i].rank == last_rank)){
                quota[cur_c]--;
                ans[cur_c].push_back(va[i].id);
                last_c =cur_c; 
                break;
            }
        }
		last_rank = va[i].rank;
    }

    for(int i=0;i<M;i++){
        sort(ans[i].begin(),ans[i].end());
        int len = ans[i].size();
        for(int j=0;j<len;j++){
            cout<<ans[i][j]<<(j<len-1?" ":"");
        }
        cout<<endl;
    }
    return 0;
}
