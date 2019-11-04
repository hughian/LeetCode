#include<bits/stdc++.h>
using namespace std;
int main()
{
	int n;
	set<int> s;
	int a[1001];
	scanf("%d",&n);
	for(int i=1;i<=n;i++){
		scanf("%d",&a[i]);
		s.insert(a[i]);
	}
	int ans=0;
	for(int i=1;i<=n;i++)
		if(s.count(a[i]+1))
			++ans;
	printf("%d\n",ans);
	return 0;	
}
