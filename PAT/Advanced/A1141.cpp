#include<iostream>
#include<cstdio>
#include<vector>
#include<algorithm>
#include<string>
#include<map>
using namespace std;
const int maxn = 100010;
struct school{
	bool valid;
	string id;
	int as,bs,ts;
	int tws;
	int ns;
	int rank;
	school():valid(false),id(""),as(0),bs(0),ts(0),tws(0),ns(0),rank(0){}
	bool operator < (const school &a)const {
		if(this->tws == a.tws){
			if(this->ns == a.ns)
				return this->id < a.id;
			return this->ns < a.ns;
		}
		return this->tws > a.tws;
	}
};
vector<school> vs(maxn);
map<string,int> idm;
int cnt = 0;
int getidx(string id){
	if(idm.count(id)==0){
		idm[id] = cnt++;
	}
	return idm[id];
}
string tolower(string &s){
	for(auto i=0;i<s.length();i++){
		if(s[i]>='A' && s[i]<='Z'){
			s[i] = s[i]-'A' + 'a';
		}
	}
	return s;
}
int main()
{
	//freopen("123.txt","r",stdin);
	int n;
	cin>>n;
	string sid,tid;
	int score;
	for(int i=0;i<n;i++){
		cin>>tid>>score>>sid;
		tolower(sid);
		int ix = getidx(sid);
		vs[ix].id = sid;
		vs[ix].valid = true;
		vs[ix].ns ++;
		if(tid[0]=='B'){
			vs[ix].bs += score;
		}else if(tid[0]=='A'){
			vs[ix].as += score;
		}else if(tid[0]=='T'){
			vs[ix].ts += score;
		}
	}
	for(int i=0;i<cnt;i++){
		if(vs[i].valid){
			vs[i].tws = vs[i].bs*2/3 + vs[i].as+vs[i].ts*3/2;
		}
	}
	sort(vs.begin(),vs.begin()+cnt);
	vs[0].rank = 1;
	for(int i=1;i<cnt;i++){
		if(vs[i].tws == vs[i-1].tws)
			vs[i].rank = vs[i-1].rank;
		else
			vs[i].rank = i+1;
	}
	cout<<cnt<<endl;
	for(int i=0;i<cnt;i++){
		printf("%d %s %d %d\n",vs[i].rank,vs[i].id.c_str(),vs[i].tws,vs[i].ns);
	}
	return 0;
}
