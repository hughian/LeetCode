#include<iostream>
#include<string>
#include<map>
#include<set>
#include<vector>
#include<cmath>
using namespace std;
vector<string> ivec;
map<string,int> imap;
//ID有可能出现 0000 与 -0000 所以应当使用string存储。
int getindex(string id){
    if(imap.count(id)==0){
        imap[id] = ivec.size();
        ivec.push_back(id);
    }
    return imap[id];
}
string getid(int index){
    return (index >= (int)ivec.size())?"":ivec[index];
}

bool isSameSign(string s,string t)
{
	//return s.length()==t.length(); //ID是四位数字加符号，长度一致时表示性别相同
    int sb = (s[0]=='-')?-1:1;
    int tb = (t[0]=='-')?-1:1;
    return (sb * tb > 0);
}
string tri(string str)
{
    if(str[0]=='-')
        str.erase(str.begin());
    return str;
}

void crush(string a,string b,int N,vector< vector<int> > &edge)
{
    set< pair<string,string> > setp;
    int ai = getindex(a);
    int bi = getindex(b);
    for(int i=0;i<N;i++){
        string af =  getid(i); 
        //选出的是a的同性朋友且不能是对象b
        if(edge[ai][i]==1 && isSameSign(af,a) && af != b){
            for(int j=0;j<N;j++){
                string bf = getid(j);
                //选出的是b的同性朋友且不能是对象a
                if(edge[bi][j]==1 && isSameSign(bf,b) && bf != a){
                    if(edge[i][j]==1)
                        setp.insert(pair<string,string>(tri(af),tri(bf)));
                }
            }
        }
    }
    cout<<setp.size()<<endl;
    for(set<pair<string,string> >::iterator it = setp.begin();it!=setp.end();it++){
        cout<<(*it).first<<" "<<(*it).second<<endl;
    }
}



int main()
{
    int N,M,K;
    cin>>N>>M;
    vector< vector<int> > edge(N,vector<int>(N,0));
    string a,b;
    int m,n;
    for(int i=0;i<M;i++){
        cin>>a>>b;
        m = getindex(a);
        n = getindex(b);
        edge[m][n] = 1;
        edge[n][m] = 1;
    }
    /*
    cout<<"______________________________________________"<<endl;
    for(int i=0;i<ivec.size();i++)
        cout<<ivec[i]<<" ";
    cout<<endl;
    cout<<"-------------"<<endl;
    for(map<int,int>::iterator it = imap.begin();it!= imap.end();it++){
        cout<<"map["<<(*it).first<<"] = "<<(*it).second<<endl;
    }
    cout<<"-------------"<<endl;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++)
            cout<<edge[i][j]<<" ";
        cout<<endl;
    }

    cout<<"______________________________________________"<<endl;
    */
    cin>>K;
    for(int i=0;i<K;i++){
        cin>>a>>b;
        crush(a,b,N,edge);
    }
    return 0;
}
