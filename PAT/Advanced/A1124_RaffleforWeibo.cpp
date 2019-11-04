#include<iostream>
#include<vector>
#include<string>
#include<map>
using namespace std;
map<string,int> mp;

int main()
{
    int m,n,s;
    cin>>m>>n>>s;
    string name;
    vector<string> vs;
    vs.push_back(" ");
    for(int i=1;i<=m;i++){
        cin>>name;
        vs.push_back(name);
    }
    if((int)vs.size() - 1 < s){ //下标从1开始
        cout<<"Keep going...";
    }else{
        int i = s;
        while(i<=m){
            if(mp.count(vs[i])==0){
                cout<<vs[i]<<endl;
                mp[vs[i]] = 1;
                i += n;
            }
            else
                i++;
        }
    }
    return 0;
}
