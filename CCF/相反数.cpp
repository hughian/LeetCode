#include<bits/stdc++.h>
using namespace std;
int main()
{
	map<int,int> m;
	int n,x,mx=0;
	scanf("%d",&n);
	for(int i=1;i<=n;i++){
		scanf("%d",&x);
		m[x]++;
		mx=max(mx,x);
	}
	int ans=0;
	for(int i=0;i<=mx;i++)
		ans+=min(m[i],m[-i]);
	printf("%d\n",ans);
	return 0;
}
