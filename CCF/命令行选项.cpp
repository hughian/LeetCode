#include<bits/stdc++.h>
using namespace std;
map<char,bool> m;
string blank(""),input;
int n;

bool checkFail(const string &b){
	int blen=b.size();
	for(int i=0;i<blen;i++)
		if(!(islower(b[i])||isdigit(b[i])||b[i]=='-')) return true;
	return false;
}

void SOLVE()
{
	static int kase=0;
	cout<<"Case "<<(++kase)<<": ";
	
	stringstream ss(input);
	string a,b;ss>>a;
	map<char,string> arg;
	while(ss>>a){
		if(a[0]!='-'||m.count(a[1])==0) break;
		else if(m[a[1]]==false) arg[a[1]]=blank;
		else{
			ss>>b;
			if(checkFail(b)) break;
			arg[a[1]]=b;
		}
	}
	
	char ch;string s;
	for(map<char,string>::iterator iter=arg.begin();iter!=arg.end();){
		ch=iter->first;s=iter->second;
		if(m[ch]) cout<<'-'<<ch<<" "<<s;
		else cout<<'-'<<ch;
		if(++iter!=arg.end()) cout<<" ";
	}

	//while(ss>>a) cout<<a<<endl;
	cout<<endl;
}

inline void INPUT(){
	char ch=cin.get();
	if(ch!='\n') cin.putback(ch);
	getline(cin,input);
}

int PREWORK()
{
	string s;
	cin>>s;
	int slen=s.size();
	for(int i=0;i<slen;i++)
		if(isalpha(s[i]))
			if(i+1<slen&&s[i+1]==':') m[s[i]]=true;
			else m[s[i]]=false;
		
//	for(map<char,bool>::iterator iter=m.begin();iter!=m.end();iter++) cout<<iter->first<<endl;
	int ret;
	cin>>ret;
	return ret;
}

void MAIN()
{
	int N=PREWORK();
	while(N--) INPUT(),SOLVE();
}

int main()
{
	//freopen("in#CCF.txt","r",stdin);
	//freopen("out#CCF.txt","w",stdout);
	MAIN();
	return 0;
}
