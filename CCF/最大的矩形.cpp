#include<bits/stdc++.h>
#define maxn 1002
using namespace std;
stack<int> s;
int a[maxn],l[maxn],r[maxn];
int main()
{
	int n;
	scanf("%d",&n);
	for(int i=1;i<=n;i++) scanf("%d",&a[i]);
	a[0]=a[n+1]=-1;
	int x;
	s.push(0);
	for(int i=1;i<=n;i++){
		for(x=s.top();a[i]<=a[x];x=s.top()) s.pop();
		l[i]=x+1;
		s.push(i);
	}
	while(!s.empty()) s.pop();
	s.push(n+1);
	for(int i=n;i>=1;i--){
		for(x=s.top();a[i]<=a[x];x=s.top()) s.pop();
		r[i]=x-1;
		s.push(i);
	}
	int ans=0;
	for(int i=1;i<=n;i++) ans=max(ans,(r[i]-l[i]+1)*a[i]);
	printf("%d\n",ans);
	return 0;
}
