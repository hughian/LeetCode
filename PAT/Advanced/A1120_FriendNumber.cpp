#include<iostream>
#include<algorithm>
#include<set>
using namespace std;
int sod(int t)
{
    int sum = 0;
    while(t){
        sum += t%10;
        t = t/10;
    }
    return sum;
}
int main()
{
    int n,tmp;
    set<int> ans;
    cin>>n;
    for(int i=0;i<n;i++){
        cin>>tmp;
        ans.insert(sod(tmp));
    }
    int num = ans.size();
    cout<<num<<endl;
    set<int>::iterator it = ans.begin();
    for(;it!=ans.end();it++){
        cout<<(*it);
        if(num>1)
            cout<<" ";
        num--;
    }
    return 0;
}
