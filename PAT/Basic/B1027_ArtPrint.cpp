#include<iostream>
#include<vector>
#include<cmath>
using namespace std;

int main()
{
    vector<int> v(1001,0);
    int tmp;
    for(int i=1;i<23;i++){
        tmp =2 * i * i - 1;
        v[tmp] = tmp;
    }
    for(int i=1;i<1001;i++){
        if(v[i]!=0)
            tmp = v[i];
        else
            v[i] = tmp;
    }
    int n;
    char c;
    cin>>n>>c;
    tmp = v[n];
    int f =2 * sqrt((tmp+1)/2)-1;
    int space = 0;
    int cnum = f;

    for(;cnum>=1;cnum-=2,space++){
        for(int i=0;i<space;i++)
            cout<<" ";
        for(int i=0;i<cnum;i++)
            cout<<c;
        cout<<endl;
    }
	cnum+=4;
	space-=2;
    for(;cnum<=f;cnum+=2,space--){
        for(int i=0;i<space;i++)
            cout<<" ";
        for(int i=0;i<cnum;i++)
            cout<<c;
        cout<<endl;
    }
    cout<<n-tmp;
    return 0;
}
