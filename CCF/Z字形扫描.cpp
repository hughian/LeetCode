#include<bits/stdc++.h>
#define maxn 500
using namespace std;
int a[maxn][maxn];
int main()
{
	//freopen("in#CCF.txt","r",stdin);
	int n;
	scanf("%d",&n);
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			scanf("%d",&a[i][j]);
	int row,col,pass,i,j;
	row=col=pass=0;
	int ans[maxn*maxn+1],k=0;
	for(row=col=0;row!=n;row++,col++,pass++){
		if(pass%2) for(i=0,j=col;j!=-1;i++,j--) ans[k++]=a[i][j];//zuo xia
		else for(i=row,j=0;i!=-1;i--,j++) ans[k++]=a[i][j];//zuo shang
	}
	for(row=col=1;row!=n;row++,col++,pass++){
		if(pass%2) for(i=row,j=n-1;i<n;i++,j--) ans[k++]=a[i][j];//zuo xia
		else for(i=n-1,j=col;j<n;i--,j++) ans[k++]=a[i][j];
	}
	for(i=0;i<k;i++) printf("%d%c",ans[i],(i==k-1)?'\n':' ');
	return 0;
}
