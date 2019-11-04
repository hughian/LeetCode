#include<iostream>
#include<vector>
using namespace std;
int diff(int a,int b)
{
    return a>b?a-b:b-a;
}
void check(vector<int>& v){
    vector<bool> used(1001,false);
    used[v[0]] = true;
    bool flag = true;
    for(int i=1;i<(int)v.size();i++){
        if(used[v[i]]==true){ //不能在同一行
            flag = false;break;
        }else{
            for(int j=0;j<i;j++){ //不能在同一斜线
                if(diff(v[j],v[i])==i-j){
                    flag = false;break;
                }
            }
            used[v[i]] = true;
        }
    }
    if(flag)
        cout<<"YES"<<endl;
    else
        cout<<"NO"<<endl;
}

int main()
{
    int k,n,t;
    vector<int> v;
    cin>>k;
    for(int i=0;i<k;i++){
        cin>>n;
        v.clear();
        for(int j=0;j<n;j++){
            cin>>t;
            v.push_back(t);
        }
        check(v);
    }
    return 0;
}
