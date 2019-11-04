#include<iostream>
#include<string>
#include<cstring>
using namespace std;
//comment this macro to use faster solve(N2)
//#define _N3_

int a[1001];
int m[1001][1001];
#ifndef _N3_
int p[1001][1001];
#endif
int s[1001];
class Solution{
	inline int min(int a,int b){
		return a<b?a:b;
	}
public:
	void ComEncode(){
		int n;
		cin>>n;
		for(int i=1;i<=n;i++)
			cin>>a[i];
		memset(m,0x08080808,sizeof(m));
		#ifndef _N3_
		memset(p,0,sizeof(p));
		#endif
		s[0] = 0;
		for(int i=1;i<=n;i++){
			s[i] = s[i-1] + a[i];
			m[i][i] = 0;
			#ifndef _N3_
			p[i][i] = i;
			#endif
		}
		for(int len =2;len<=n;len++){
			for(int i=1;i+len-1<=n;i++){
				int j = i+len-1;
				#ifndef _N3_
				#define _N2_
				for(int k=p[i][j-1];k<=p[i+1][j];k++){
					int tmp = m[i][k] + m[k+1][j] + s[j] - s[i-1];
					if(tmp < m[i][j]){
						m[i][j] = tmp;
						p[i][j] =k;
					}
				}
				#endif
				#ifndef _N2_
				for(int k=i;k<j;k++){
					m[i][j] = min(m[i][j],m[i][k]+m[k+1][j]+s[j]-s[i-1]);
				}
				#endif
			}
		}
		cout<<m[1][n]<<endl;
	}
};

int main()
{
	Solution s;
	s.ComEncode();
	return 0;
}