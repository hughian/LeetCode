#include<iostream>
#include<vector>
using namespace std;
vector< vector<int> > cost(520,vector<int>(520,-1));
vector<int> team(520,0);
vector<bool> used(520,false);
vector<int> ans;
int N,M,st,dt;
int res = 0;
int dist=0,tmp=0,maxt=0,mind = 1e9;
bool flag = false;
vector<int> di,te;
void _dfs(int v){
	tmp += team[v];
	if(v==dt){
		di.push_back(dist);
		te.push_back(tmp);
	}
	for(int i=0;i<N;i++){
		if(!used[i] && cost[v][i] != -1){
			used[i] = true;
			dist += cost[v][i];
			_dfs(i);
			dist -= cost[v][i];
			used[i] =false;
		}
	}
	tmp -= team[v];
}

int main()
{
    cin>>N>>M>>st>>dt;
    for(int i=0;i<N;i++) cin>>team[i];
    int a,b,L;
    for(int i=0;i<M;i++){
        cin>>a>>b>>L;
        cost[a][b] = cost[b][a] = L;
    }
    used[st] = true;
    _dfs(st);
	for(int i=0;i<(int)di.size();i++){
		if(di[i]<mind)
			mind = di[i];
	}
	for(int i=0;i<(int)di.size();i++){
		if(di[i]==mind){
			res++;
			if(te[i] > maxt)
				maxt = te[i];
		}
		
	}
    cout<<res<<" "<<maxt<<endl;
    return 0;
}
