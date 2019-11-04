#include<iostream>
#include<string>
#include<vector>
using namespace std;

bool check(string s){
    int len = s.length();
    int r = s.find("PAT");
    if(r!=string::npos){
        string left = s.substr(0,r);
        string right= s.substr(r+3,len-r-3);
        return (left==right);
    }else{
        int ppos = s.find('P');
        int tpos = s.find('T');
        string a = s.substr(0,ppos);
        string ca= s.substr(tpos+1,len-tpos-1);
        string b = s.substr(ppos+1,tpos-ppos-1);
		return (ca.length()==a.length()*b.length());
    }
}


int main()
{
    int n;
    cin>>n;
    string s;
    for(int i=0;i<n;i++){
        cin>>s;
        vector<int> cnt(3,0);
        for(unsigned i=0;i<s.length();i++){
            if(s[i]=='P') cnt[0]++;
            else if(s[i]=='A') cnt[1]++;
            else if(s[i]=='T') cnt[2]++;
        }
        if(s.find_first_not_of("PAT") != string::npos)
            cout<<"NO\n";
        else if(s.find('P')==string::npos || s.find('A')==string::npos || s.find('T')==string::npos)
            cout<<"NO\n";
        else{
            if(cnt[0]>1 || cnt[2]>1)
                cout<<"NO\n";
            else
                cout<<(check(s)?"YES\n":"NO\n");
        }
    }
    return 0;
}
