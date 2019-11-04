#include<iostream>
#include<string>
#include<algorithm>
#include<map>
using namespace std;
typedef long long ll;
ll Inf =((1LL<<63)-1LL);
map<char,ll> mp;
int init()
{
    for(char c='0';c<='9';c++)
        mp[c] = c-'0';
    for(char c='a';c<='z';c++)
        mp[c] = c - 'a' + 10;
}
ll trans(string str,ll base = 10,ll t = Inf)
{
	ll sum = 0;
    int len = str.length();
    for(int i=0;i<len;i++){
        sum = sum * base + mp[str[i]];
        if(sum <0 || sum > t) return -1; //溢出或者大于t
    }
	return sum;
}
ll findMaxDigit(string str){
    int len = str.length();
    ll max = -1;
    for(int i=0;i<len;i++){
        if(mp[str[i]] > max)
            max = mp[str[i]];
    }
    return max+1;
}
int cmp(string str,ll base,ll t1){
    ll t2 = trans(str,base,t1);
    if(t2 < 0) return 1;//t2在转换过程中发生了溢出，或者结果已经大于了t1
    else if(t2 < t1) return -1;
    else if(t2 == t1) return 0; 
    else return 1;
}

ll binarySearch(ll low,ll high,string &str,ll n1)
{
    ll mid;
    while(low<=high){
        mid = (low+high)/2;
        int r = cmp(str,mid,n1);
        if(r==-1) low = mid+1;
        else if(r==1) high = mid -1;
        else return mid;
    }
    return -1;
}

int main()
{
    init();
    string n1,n2;
    ll tag,radix;
    cin>>n1>>n2>>tag>>radix;
    if(tag == 2){
        string tmp = n1;
        n1 = n2;
        n2 = tmp;
    }
    ll t1 = trans(n1,radix);
    ll low = findMaxDigit(n2);
    ll high = max(low,t1)+1;
    ll ans = binarySearch(low,high,n2,t1);
    if(ans!=-1)
        cout<<ans;
    else
        cout<<"Impossible";
    return 0;
}
