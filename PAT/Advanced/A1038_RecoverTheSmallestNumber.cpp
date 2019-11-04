#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
using namespace std;
vector<string> vs(10010);

bool cmp(string &a,string &b){
    return a+b < b+a;
}

int main()
{
    int N;
    cin>>N;
    for(int i=0;i<N;i++) cin>>vs[i];
    sort(vs.begin(),vs.begin()+N,cmp);
    string ans;
    for(unsigned i=0;i<vs.size();i++)
        ans += vs[i];
    while(ans[0]=='0') ans.erase(ans.begin());
	if(ans.size()==0) //结果为0的时候输出一个0
		cout<<0;
	else
		cout<<ans;
    return 0;
}
