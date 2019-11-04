#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
using namespace std;
string str;
vector< vector<int> > vvi;
int lcs(string a,string b)
{
	int max = -1;
	vector< vector<int> >  vvi(1001,vector<int>(1001,0));
	unsigned i ,j;
	for(i=0;i<a.length();i++){
		for(j=0;j<b.length();j++){
			if(a[i]==b[j]){
				if(i>0 && j>0)
					vvi[i][j] = vvi[i-1][j-1]+1;
				else
					vvi[i][j] = 1;
				if(max < vvi[i][j])
					max = vvi[i][j];
			}
		}
		
	}
	return max;
}

int main()
{
    getline(cin,str);
    string tmp = str;
    reverse(tmp.begin(),tmp.end());
    cout<<lcs(str,tmp);
	return 0;
}
