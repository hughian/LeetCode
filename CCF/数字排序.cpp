#include<bits/stdc++.h>
#define maxn 1001
using namespace std;
struct Node{
	int val,times;
	Node(int val,int times):val(val),times(times){}
	inline void print()const{printf("%d %d\n",val,times);}
	bool operator <(const Node &b)const{return times<b.times||(times==b.times&&val>b.val);}
};
int main()
{
	int n,in;
	scanf("%d",&n);
	map<int,int> m;
	for(int i=0;i<n;i++){
		scanf("%d",&in);
		m[in]++;
	}
	priority_queue<Node> p;
	for(map<int,int>::iterator iter=m.begin();iter!=m.end();iter++) p.push(Node(iter->first,iter->second));
	while(!p.empty()){
		p.top().print();p.pop();
	}
	return 0;
}
