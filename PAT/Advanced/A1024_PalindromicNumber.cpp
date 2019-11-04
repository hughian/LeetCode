#include<iostream>
#include<string>
#include<algorithm>
using namespace std;
bool isPalin(string str){
    int i=0,j=str.length()-1;
    while(i<=j){
        if(str[i]!=str[j])
            return false;
        else{
            i++;j--;
        }
    }
    return true;
}
string add(string a,string b){
    int i = a.length()-1;
    int j = b.length()-1;
    int t,c=0;
    string ans;
    while(i>=0 && j>=0){
        t = a[i]-'0' + b[j]-'0' + c;
        ans.push_back(t%10+'0');
        c =  t/10;
        i--;j--;
    }
    while(i>=0){
        t = a[i]-'0' + c;
        ans.push_back(t%10+'0');
        c = t/10;
        i--;
    }
    while(j>=0){
        t = b[j]-'0' +c;
        ans.push_back(t%10+'0');
        c = t/10;
        j--;
    }
    if(c)
        ans.push_back(c%10+'0');
    reverse(ans.begin(),ans.end());
    return ans;
}
int main()
{
    string n;
    int k;
    cin>>n>>k;
    int t = 0;
	string st;
	while(!isPalin(n)){
        st = n;
        reverse(st.begin(),st.end());
        n = add(st,n);
        t++;
        if(t==k)
            break;
    }
    cout<<n<<endl<<t<<endl;
    return 0;
}
