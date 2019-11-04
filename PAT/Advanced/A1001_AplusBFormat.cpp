#include<iostream>
#include<cstdio>
using namespace std;

/*
//另外一种思路
#include<iostream>
#include<cmath>
#include<stack>
using namespace std;

int main()
{
    long long a,b,c;
    cin>>a>>b;
    c = a+b;
    long long x = fabs(c);
    stack<char> s;
    int i=0;
    while(x){
        if(i==3){
            s.push(',');
            i=0;
        }
        s.push((x%10+'0'));
        x = x/10;
        i++;
    }
    if(c<0)
        s.push('-');
	else if(c==0) //0的情况要单独考虑
		s.push('0');
	
    while(!s.empty()){
        char c = s.top();s.pop();
        cout<<c;
    }
    return 0;
}

*/


int main()
{
    int a,b,c;
	cin>>a>>b;
	c = a + b;
	
	if(c>-1000 && c<1000)
		printf("%d",c);
	else if(c > -1000000 && c<=-1000){
		printf("%d,",c/1000);
		printf("%03d",(-c)%1000);
	}
	else if(c <= -1000000){
		printf("%d,",c/1000000);
		printf("%03d,",((-c)/1000)%1000);
		printf("%03d",(-c)%1000);
	}
	else if(c >= 1000 && c< 1000000){
		printf("%d,",c/1000);
		printf("%03d",c%1000);
	}
	else {
		printf("%d,",c/1000000);
		printf("%03d,",(c/1000)%1000);
		printf("%03d",c%1000);
	}
	return 0;
}
