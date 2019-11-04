#include<bits/stdc++.h>
using namespace std;
int main()
{
	ios::sync_with_stdio(false);
	map<int,int> m;
	int n,in;
	cin>>n;
	while(n--){
		cin>>in;
		cout<<(++m[in])<<(n==0?'\n':' ');
	}
	return 0;
}
