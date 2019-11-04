#include<iostream>
#include<vector>
#include<cstdio>
#include<queue>
using namespace std;

vector< vector<int> > children(101);
vector<int> parent(101,-1);
int N,M;

void lvOrder(int root)
{
    queue<int> q;
    q.push(root);
    q.push(-1);
    int max =0,cnt=0;
    int lv = 0,maxlv;
    while(q.size()>1){
        int t = q.front();q.pop();
        if(t==-1){
            q.push(-1);
            if(cnt>max){
                max = cnt;
				maxlv = lv;
			}
            lv++;
            cnt = 0;
        }else{
            cnt++;
            for(int i=0;i<(int)children[t].size();i++)
                q.push(children[t][i]);
        }
    }
	if(cnt>max){ //一定不要忽略循环结束后最后一层还没有判断
		max = cnt;
		maxlv = lv;
	}
    cout<<max<<" "<<maxlv+1;
}

int main()
{
    cin>>N>>M;
    int id,k,t;
    for(int i=0;i<M;i++){
        cin>>id>>k;
        for(int j=0;j<k;j++){
            cin>>t;
            children[id].push_back(t);
            parent[t] = id;
        }
    }
    lvOrder(1);
    return 0;
}
