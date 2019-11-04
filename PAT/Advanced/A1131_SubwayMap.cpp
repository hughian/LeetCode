#include<iostream>
#include<vector>
#include<queue>
#include<cstdio>
using namespace std;
int printf(const char*,...);
int scanf(const char*,...);
vector< vector<int> > edge(10001);
int line[10000][10000];
vector<bool> visit(10001,false);
vector<int> path,tempPath;
int N;

int transferCnt(vector<int> &v){
    int cnt = -1,preLine = 0;
    for(int i=1;i<(int)v.size();i++){
        if(line[v[i-1]][v[i]] != preLine ) cnt ++;
        preLine = line[v[i-1]][v[i]];
    }
    return cnt;
}

void dfs(int src,int dst,int cnt,int &minCnt,int &minTransfer)
{
    int tCnt = transferCnt(tempPath)
    if(src == dst && (cnt <minCnt || (cnt == minCnt && tCnt <minTransfer))){
        minCnt = cnt;
        minTransfer = tCnt;
        path = tempPath;
    }
    if(src == dst) return;
    for(int i=0;i<(int)edge[src].size();i++){
        if(visit[edge[src][i]] ==false){
            tempPath.push_back(edge[src][i]);
            visit[edge[src][i]] = true;
			dfs(edge[src][i],dst,cnt+1,minCnt,minTransfer);
            visit[edge[src][i]] = false;
            tempPath.pop_back();
        }
    }
}

int main()
{
    int m,k;
    int last,t;
    cin>>N; 
    for(int i=1;i<=N;i++){
        cin>>m>>last;
        for(int j=1;j<m;j++){
            cin>>t;
            edge[last].push_back(t); //ÁÚ½Ó±í
            edge[t].push_back(last);

            line[last][t] = line[t][last] = i;
            last = t;
        }
    }
    int src,dst;
    cin>>k;
    for(int i=0;i<k;i++){
        cin>>src>>dst;
        int minCnt = 10000,minTransfer=10000;
        tempPath.clear();
		
        tempPath.push_back(src);
        visit[src] = true;
        dfs(src,dst,0,minCnt,minTransfer);
        
        cout<<minCnt<<endl;
        int preLine = 0,preTransfer = src;
        for(int j=1;j<(int)path.size();j++){
            if(line[path[j-1]][path[j]] != preLine) {//»»³Ë
                if(preLine !=0)
                    printf("Take Line#%d from %04d to %04d.\n",preLine,preTransfer,path[j-1]);
                preLine = line[path[j-1]][path[j]];
                preTransfer = path[j-1];
            }
        }
        printf("Take Line#%d from %04d to %04d.\n",preLine,preTransfer,dst);
    }
    return 0;
}
