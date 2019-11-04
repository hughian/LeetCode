#include<iostream>
#include<stack>
#include<string>
#include<cstdio>
using namespace std;

string a[10] = {"ling","yi","er","san","si","wu","liu","qi","ba","jiu"};
int main()
{
	char c;
	int ans = 0;
	string str;
	int tmp;
	stack<int> s;
	cin>>str;
	for(int i=0;i<str.length();i++)
		ans += str[i]-'0';
	while(ans){
		tmp = ans%10;
		ans = ans/10;
		s.push(tmp);
	}
	tmp = s.top();s.pop();
	printf("%s",a[tmp].data());
	while(!s.empty()){
		tmp = s.top();s.pop();
		printf(" %s",a[tmp].data());
	}
	return 0;
}