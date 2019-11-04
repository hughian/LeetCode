#include<iostream>
#include<string>
using namespace std;
//注意A<B时商为0.此时只有一个"0"
int main()
{
	string str,r;
	int q=0,b,tmp;
	cin>>str>>b;
	str = "0" + str;
	char buf[2] = {'\0'};
	for(int i=1;i<str.length();i++)
	{
		tmp = q*10;
		tmp = tmp + str[i] - '0';
		
		buf[0] = tmp/b + '0';
		r += string(buf);
		
		q = tmp % b;
	}
	if(r.length()>1){
		int k = r.find_first_not_of("0");
		r = r.substr(k,r.length()-k);
	}
	cout<<r<<" "<<q;
	return 0;
}
