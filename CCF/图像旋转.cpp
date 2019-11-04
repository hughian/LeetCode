#include<bits/stdc++.h>
using namespace std;
int a[1001][1001]; 
int main()
{
	int n,m;
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)
		for(int j=1;j<=m;j++)
			scanf("%d",&a[i][j]);
	for(int i=m;i>=1;i--)
		for(int j=1;j<=n;j++)
			printf("%d%c",a[j][i],j==n?'\n':' ');
	return 0;
}
