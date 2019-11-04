#include<iostream>
#define eps 1e-9
using namespace std;
int main()
{
    double a,x,y,b,h;
    cin>>a>>x>>y>>b>>h;
	double aa = (-4*y)/(x*x);
    double yt = aa*(b-a-x/2.0)*(b-a-x/2.0) + y;
    if(b<a || b>a+x){
		cout<<"YES\n";
		return 0;
	}
	if((yt-h)<=eps)
        cout<<"NO\n";
    else
        cout<<"YES\n";
    return 0;
}
