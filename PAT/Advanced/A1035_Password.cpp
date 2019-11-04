#include<iostream>
#include<string>
#include<vector>
using namespace std;
vector< string > vp;
bool alter(string &str){
    unsigned i=0;
    bool flag = false;
    for(;i<str.length();i++){
        if(str[i]=='l'){
            str[i] = 'L';
            flag = true;
        }else if(str[i]=='O'){
            str[i] = 'o';
            flag = true;
        }else if(str[i]=='1'){
            str[i] = '@';
            flag = true;
        }else if(str[i]=='0'){
            str[i]='%';
            flag = true;
        }
    }
    return flag;
}
int main()
{
    int N;
    string id,pwd;
    cin>>N;
    for(int i=0;i<N;i++){
        cin>>id>>pwd;
        if(alter(pwd))
            vp.push_back(id+" "+pwd);
    }
    if(vp.size()==0){
        if(N==1)
			cout<<"There is "<<N<<" account and no account is modified\n";
		else
			cout<<"There are "<<N<<" accounts and no account is modified\n";
	}else{
        cout<<vp.size()<<endl;
        for(int i=0;i<(int)vp.size();i++){
            cout<<vp[i]<<endl;
        }
    }
    return 0;
}

