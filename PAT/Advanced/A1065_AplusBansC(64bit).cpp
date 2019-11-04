#include<iostream>
using namespace std;

int main()
{
	long long t,a,b,c;
    cin>>t;
    for(int i=1;i<=t;i++){
        cin>>a>>b>>c;
		long long tmp = a+b; //考察溢出判断
		if(a>0 && b>0 &&(tmp < a || tmp<b))
			cout<<"Case #"<<i<<": true"<<endl;
		else if(a<0 && b<0 && (tmp>a || tmp>b))
			cout<<"Case #"<<i<<": false"<<endl;
        else if(tmp>c)
            cout<<"Case #"<<i<<": true"<<endl;
        else
            cout<<"Case #"<<i<<": false"<<endl;
    }
    return 0;
}
