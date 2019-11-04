#include<iostream>
#include<vector>
using namespace std;

int main()
{
    int a,e;
    vector<pair<int,int> > v; 
    while(cin>>a>>e,!cin.eof()){
        pair<int,int> pt;
        if(e==0){
            continue;
        }
        else{
            pt.first = a*e;
            pt.second = e-1;
        }
        v.push_back(pt);
    }
    if(v.size()==0){
        cout<<"0 0";
        return 0;
    }
    cout<<v[0].first<<" "<<v[0].second;
    for(int i=1;i<(int)v.size();i++){
        cout<<" "<<v[i].first<<" "<<v[i].second;
    }
    return 0;
}
