#include<iostream>
#include<vector>
#include<string>
#include<map>
#include<queue>
#include<algorithm>
using namespace std;
const int Inf = 1e9;
vector< vector<int> > cost(210,vector<int>(210,-1));
vector<bool> used(210,false);
vector<int> hap(210,0);
vector<string> vsc(210);
map<string,int> mpc;
struct PATH{
    int totalCost,totalHap,avgHap;
    vector<int> path;
    PATH(int tc,int th,vector<int>&p)
        :totalCost(tc),totalHap(th),path(p){
        if(path.size()==1)
            avgHap = totalHap;
        else
            avgHap = totalHap / (path.size()-1);
    }
};
vector<int> path;
vector<PATH> ans;
int N,K;
int cnt = 0;
int st,dt;
int minC = Inf,maxH = -1;
int getidx(string city){
    if(mpc.count(city)==0){
        mpc[city] = cnt;
        vsc[cnt] = city;
        cnt++;
    }
    return mpc[city];
}
bool cmp(PATH &a,PATH &b){
    if(a.totalCost == b.totalCost){
        if(a.totalHap == b.totalHap)
            return a.avgHap > b.avgHap;
        return a.totalHap > b.totalHap;
    }
    return a.totalCost < b.totalCost;
}

void dfs(int v,int c,int h){
    if(v==dt){
        ans.push_back(PATH(c,h,path));
    }
    for(int i=0;i<N;i++){
        if(!used[i] && cost[v][i]!=-1){
            path.push_back(i);
            used[i] = true;
            dfs(i,c+cost[v][i],h+hap[i]);
            path.pop_back();
            used[i] = false;
        }
    }
}

int main()
{
    string start,end="ROM",city;
    cin>>N>>K>>start;
    st = getidx(start);
    dt = getidx(end);
    hap[st] = 0;
    int h;
    for(int i=1;i<N;i++){
        cin>>city>>h;
        int ci = getidx(city);
        hap[ci] = h;
    }
    string a,b;
    int c;
    for(int i=0;i<K;i++){
        cin>>a>>b>>c;
        int ai = getidx(a);
        int bi = getidx(b);
        cost[ai][bi] = cost[bi][ai] = c;
    }
    path.push_back(st);
    used[st] = true;
    dfs(st,0,0);

    sort(ans.begin(),ans.end(),cmp);
    int k=1;
    for(;k<(int)ans.size();k++){
        if(ans[k].totalCost != ans[k-1].totalCost)
            break;
    }
    cout<<k<<" "<<ans[0].totalCost<<" "<<ans[0].totalHap<<" "<<ans[0].avgHap<<endl;
    int len = ans[0].path.size();
    for(int i=0;i<len;i++){
        cout<<vsc[ ans[0].path[i] ];
        if(i<len-1) cout<<"->";
    }
    /*
    cout<<"\n________________________________"<<endl;
    for(int i=0;i<(int)ans.size();i++){
        cout<<ans[i].totalCost<<" "<<ans[i].totalHap<<" "<<ans[i].avgHap<<endl;
        len = ans[i].path.size();
        for(int j=0;j<len;j++){
            cout<<ans[i].path[j]<<":"<<vsc[ ans[i].path[j] ];
            if(j<len-1) cout<<"->";
        }
        cout<<endl;
    }*/
    return 0;
}
