#include<iostream>
#include<algorithm>
#include<vector>

using namespace std;
int getsum(vector<int> &v,int left,int right)
{
    int sum = 0;
    for(int i=left;i<right;i++)
        sum += v[i];
    return sum;
}
int main()
{
    int n;
    cin>>n;
    vector<int> v(n,0);
    for(int i=0;i<n;i++)
        cin>>v[i];
    sort(v.begin(),v.end());
    int s1 = 0, s2 = 0;
    if(n%2==0){
        s1 = getsum(v,0,n/2);
        s2 = getsum(v,n/2,n);
        cout<<"0 "<<(s2-s1);
    }else{
        s1 = getsum(v,0,n/2);
        s2 = getsum(v,n/2,n);
        int t1 = s2-s1;
        s1 = getsum(v,0,n/2+1);
        s2 = getsum(v,n/2+1,n);
        int t2 = s2-s1;
        cout<<"1 "<<(t1>t2?t1:t2);
    }
    return 0;
}
