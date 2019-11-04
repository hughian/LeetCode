#include<iostream>
#include<vector>
using namespace std;
int main()
{
    int n,b;
    cin>>n>>b;
    vector<int> v;
    if(n==0){
        cout<<"Yes\n0";
        return 0;
    }
    while(n){
        v.push_back(n%b);
        n = n/b;
    }
    int i=0,j=v.size()-1;
    bool flag = true;
    while(i<=j){
        if(v[i]!=v[j]){
            flag = false;
            break;
        }else{
            i++;j--;
        }
    }
    cout<<(flag?"Yes\n":"No\n");
    for(i=v.size()-1;i>=0;i--){
        cout<<v[i];
        if(i>0) cout<<" ";
    }
    return 0;
}

