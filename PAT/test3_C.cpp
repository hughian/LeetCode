#include<iostream>
#include<vector>
#include<string>
#include<map>
#include<algorithm>
using namespace std;
vector<int> ufs(1e4+10,-1);
vector<int> idx(1e4+10,0);
vector<bool> isLeaf(1e4+10,true);
map<int,int> mp;
int N,M;

string gettype(int i,int &ix){
    int t = i;
    string res;
    while(ufs[t]!=-1){
        res.push_back(idx[t]+'0');
        t = ufs[t];
    }
    ix = mp[t];
    res.push_back(idx[t]+'0');
    return res;
}
string add(const string &a,const string &b)
{
    int i=0,j=0;
    int lena = a.length();
    int lenb = b.length();
    int t =0,c = 0;
    string ans;
    while(i<lena && j<lenb){
        t = a[i] -'0' + b[j]-'0' +c;
        ans.push_back(t%10+'0');
        c = t/10;
        i++;j++;
    }
    while(i<lena){
        t = a[i] -'0' +c;
        ans.push_back(t%10+'0');
        c = t/10;
        i++;
    }
    while(j<lenb){
        t = b[j] -'0' +c;
        ans.push_back(t%10+'0');
        c = t/10;
        j++;
    }
    if(c)
        ans.push_back(c+'0');
    return ans;
}
void trim(string &s){
    while(s[0]=='0') s.erase(s.begin());
    if(s.size()==0)
        s.push_back('0');
}
void sum(vector<string> &vs){
    for(unsigned i=1;i<vs.size();i++){
        vs[0] = add(vs[0],vs[i]);
    }
    reverse(vs[0].begin(),vs[0].end());
    trim(vs[0]);
}
int main()
{
    cin>>N>>M;
    for(int i=0;i<N;i++)
        cin>>idx[i];
    int a,b,k=0,ix;
    for(int i=0;i<M;i++){
        cin>>a>>b;
        isLeaf[a] = false;
        ufs[b] = a;
    }
    for(int i=0;i<N;i++){
        if(ufs[i]==-1){
            mp[i] = k;
            k++;
        }
    }
    vector< vector<string> > ans(k+1);
    for(int i=0;i<N;i++){
        if(isLeaf[i]){
            string s = gettype(i,ix);
            ans[ix].push_back(s);
        }
    }
    cout<<k<<endl;
    for(int i=0;i<k;i++){
        sum(ans[i]);
        cout<<ans[i][0];
        if(i<k-1) cout<<" ";
    }

    return 0;
}
