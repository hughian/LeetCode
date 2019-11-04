#include<iostream>
#include<vector>
#include<cstdio>
#include<stack>
using namespace std;
const int Inf = 1e9 + 1;
vector< vector<int> > cost(520,vector<int>(520,Inf));
vector< vector<int> > tttt(520,vector<int>(520,Inf));
int N,M;
vector<int> path(520,-1);
vector<int> dist(520,Inf);
vector<int> tm(520,Inf);
vector<int> ditm(520,Inf);
vector<int> tpt(520,-1);
vector<int> cross(520,Inf);
void dijstrka(int v){
    vector<bool> used(520,false);
    vector<bool> tvis(520,false);
    for(int i=0;i<N;i++){
        dist[i] = cost[v][i];
        ditm[i] = tttt[v][i];
        tm[i] = tttt[v][i];
        path[i] = v;
        tpt[i] = v;
    }
    dist[v] = 0;
    ditm[v] = 0;
    tm[v] = 0;
    used[v] = true;
    tvis[v] = true;
    cross[v] = 0;

    int min = Inf;
    int w,u;
    for(int i=1;i<N;i++){
        min = Inf;
        for(int j=0;j<N;j++){
            if(!used[j] && min > dist[j]){
                    min = dist[j];
                    u = j;
            }
        }
        min = Inf;
        for(int j=0;j<N;j++){
            if(!tvis[j] && min>ditm[j]){
                min = ditm[j];
                w = j;
            }
        }

        used[u] = true;
        tvis[w] = true;

        for(int j=0;j<N;j++){
            if(!used[j] ){
                if(dist[j] > dist[u]+cost[u][j]){
                    dist[j] = dist[u]+cost[u][j];
                    tm[j] = tm[u] + tttt[u][j];
                    path[j] = u;
                }else if(dist[j]==dist[u]+cost[u][j] && tm[j] > tm[u]+tttt[u][j]){
                    tm[j] = tm[u] + tttt[u][j];
                    path[j] = u;
                }
            }
            if(!tvis[j] && ditm[j] > ditm[w] + tttt[w][j]){
                ditm[j] = ditm[w] + tttt[w][j];
                cross[j] = cross[w] + 1;
                tpt[j] = w;
            }else if(!tvis[j] && ditm[j] == ditm[w] + tttt[w][j] && cross[j]>cross[w]+1){
                cross[j] = cross[w] + 1;
                tpt[j] = w;
            }
        }
    }
}

void printStk(stack<int>&s)
{
    while(!s.empty()){
        int t= s.top();s.pop();
        cout<<t;
        if(s.size()>0) cout<<" -> ";
    }
}

int main()
{
    cin>>N>>M;
    int a,b,oneway,len,t;
    for(int i=0;i<M;i++){
        cin>>a>>b>>oneway>>len>>t;
        cost[a][b] = len;
        tttt[a][b] = t;
        if(oneway==0){
            cost[b][a] = len;
            tttt[b][a] = t;
        }
    }
    int st,dt;
    cin>>st>>dt;
    dijstrka(st);
    stack<int> sdist,stime;
    int p = dt;
    t = dt;
    bool flag = true;
    while(p!=st && t!=st){
        if(p!=t)
            flag = false;
        sdist.push(p);
        stime.push(t);
        p = path[p];
        t = tpt[t];
    }
    while(p!=st){
        flag = false;
        sdist.push(p);
        p = path[p];
    }
    while(t!=st){
        flag = false;
        stime.push(t);
        t = tpt[t];
    }
    sdist.push(st);
    stime.push(st);
    if(flag){
        cout<<"Distance = "<<dist[dt]<<"; Time = "<<ditm[dt]<<": ";
        printStk(sdist);
    }else{
        cout<<"Distance = "<<dist[dt]<<": ";
        printStk(sdist);
        cout<<endl;
        cout<<"Time = "<<ditm[dt]<<": ";
        printStk(stime);
    }
    cout<<endl;
    return 0;
}
