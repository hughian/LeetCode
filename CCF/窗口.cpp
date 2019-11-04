#include<bits/stdc++.h>
using namespace std;
struct Node{
	int x1,y1,x2,y2;
	int x,y;
	int no;
	void read(int i){scanf("%d%d%d%d",&x1,&y1,&x2,&y2);no=i;}
	void read_c(){scanf("%d%d",&x,&y);}
	bool operator ==(Node b)const{return (x1<=b.x&&b.x<=x2&&y1<=b.y&&b.y<=y2);}
};
deque<Node> d;
typedef deque<Node>::iterator iter;
int main()
{
//	freopen("in#CCF.txt","r",stdin);
	int n,m;
	Node x;
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++){
		x.read(i);
		d.push_front(x);
	}
	for(int i=1;i<=m;i++){
		x.read_c();
		iter it=find(d.begin(),d.end(),x);
		if(it==d.end()) printf("IGNORED\n");
		else{
			x=*it;
			printf("%d\n",x.no);
			d.erase(it);
			d.push_front(x);
		}		
	}
	return 0;
}
