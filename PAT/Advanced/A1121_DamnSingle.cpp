#include<iostream>
#include<vector>
#include<algorithm>
#include<set>
#include<cstdio>
using namespace std;
vector<int> couple(100001,-1);

int main()
{
    int n,m;
    cin>>n;
    int a,b;
    set<int> pat;
    for(int i=0;i<n;i++){
        cin>>a>>b;
        couple[a] = b;
        couple[b] = a;
    }
    cin>>m;
    for(int i=0;i<m;i++){
        cin>>a;
        pat.insert(a);
    }
    set<int>::iterator it = pat.begin();
    vector<int> ans;
    for(;it!=pat.end();it++){
        if(couple[(*it)]==-1)
            ans.push_back((*it));
        else if(pat.count(couple[(*it)])==0)
            ans.push_back((*it));
    }
    cout<<ans.size()<<endl;
    for(unsigned i=0;i<ans.size();i++){
        printf("%05d",ans[i]); //Êä³öÒªÌî³ä
        if(i<ans.size()-1)
            cout<<" ";
    }
    return 0;
}
