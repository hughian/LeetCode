#include<iostream>
#include<algorithm>
#include<vector>
#include<cstdio>
#include<queue>
using namespace std;
int line[10000][10000];
vector<int> arc[10000];
vector<bool> used(10000,false);
vector<int> path,tmpPath;
int N;
int src,dst;
int minCnt,minTrans;

int transCount(vector<int>&v){
	int cnt = -1,preLine = 0;
	for(int i=1;i<(int)v.size();i++){
		int last = v[i-1];
		int t = v[i];
		if(line[last][t] != preLine){
			cnt++;
		}
		preLine = line[last][t];
	}
	
}


void dfs(int v,int cnt){
	int tmpCnt = transCount(tmpPath);
	if(v == dst){
		if(cnt < minCnt || (cnt == minCnt && tmpCnt < minTrans)){
			minCnt = cnt;
			minTrans = tmpCnt;
			path = tmpPath;
		}
		return;
	}
	for(int i=0;i<(int)arc[v].size();i++){
		int ix = arc[v][i];
		if(!used[ix]){
			tmpPath.push_back(ix);
			used[ix] = true;
			dfs(ix,cnt+1);
			tmpPath.pop_back();
			used[ix] = false;
		}
	}
}

int main(){
	int m,k;
	int last,t;
	cin>>N;
	for(int i=1;i<=N;i++){
		cin>>m>>last;
		for(int j=1;j<m;j++){
			cin>>t;
			arc[last].push_back(t);
			arc[t].push_back(last);
			
			line[last][t] = line[t][last] = i;
			last = t;
		}
	}
	cin>>k;
	for(int i=0;i<k;i++){
		cin>>src>>dst;
		minCnt = 10000;
		minTrans=10000;
		tmpPath.clear();
		tmpPath.push_back(src);
		fill(used.begin(),used.end(),false);
		used[src] = true;
		dfs(src,0);
	}
}