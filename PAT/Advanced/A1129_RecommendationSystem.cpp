#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<map>
#include<set>
using namespace std;

bool cmp(pair<int,int> &a,pair<int,int> &b){
    if(a.second == b.second){
        return a.first < b.first; 
    }
    return a.second > b.second;
}

void print(vector<pair<int,int> >& vp){
    for(unsigned i=0;i<vp.size();i++){
        cout<<"("<<vp[i].first<<":"<<vp[i].second<<") ";
    }
    cout<<endl;
}

int main()
{
    int N,K;
    cin>>N>>K;
    int last,t;
    map<int,int> mp;
    vector<pair<int,int> > ans;
    cin>>last;
    for(int i=1;i<N;i++){
        cin>>t;
        mp[last]++;
        bool flag = false;
        int len = ans.size();
        for(int i=0;i<len;i++)
            if(last == ans[i].first){
                ans[i].second = mp[last];
                flag = true;
            }
        if(!flag){
            if(len >= K){
                if(ans[len-1].second < mp[last] || \ 
							(ans[len-1].second ==mp[last] && last< ans[len-1].first))
                    ans[ans.size()-1] = make_pair(last,mp[last]);
            }else{
                ans.push_back(make_pair(last,mp[last]));
            }
        }
        sort(ans.begin(),ans.end(),cmp);
        cout<<t<<":";
        for(unsigned i=0;i<ans.size();i++){
            cout<<" "<<ans[i].first;
        }
        cout<<endl;
        last = t;
    }
    return 0;
}
