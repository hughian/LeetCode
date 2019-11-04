#include<iostream>
#include<vector>
#include<set>
using namespace std;
vector< vector<int> > mat(110,vector<int>(110,0));
int main()
{
    int k,t;
    set<int> v;
    cin>>k;
    for(int i=0;i<k;i++){
        cin>>t;
        v.insert(t);
    }
    set<int>::iterator it = v.begin();
    for(;it!=v.end();it++){
        t = *it;
        while(t!=1){
            if(t%2==0) t/=2;
            else t = (3*t+1)/2;
            if(v.count(t)==1)
                v.erase(t);
        }
    }
    int len = v.size(),cnt=0;
    set<int>::reverse_iterator rit = v.rbegin();
    for(;rit!=v.rend();rit++){
        cout<<*rit;
        if(cnt<len-1) cout<<" ";
        cnt++;
    }
    return 0;
}
