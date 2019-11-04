#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
using namespace std;
string str[]={"S","H","C","D"};
struct Node{
    string str;
    int idx;
    bool operator < (const Node &a)const{
        return this->idx < a.idx;
    }
};
vector<Node> vn(60);
vector<int> vi(60);
int main()
{
    int k;
    cin>>k;
    for(int i=1;i<=54;i++) cin>>vi[i];
    for(int i=0;i<4;i++){
        for(int j=1;j<=13;j++){
            vn[i*13+j].str = str[i] + to_string(j);
        }
    }
    vn[53].str = "J1";vn[54].str = "J2";

    while(k--){
        for(int i=1;i<=54;i++){
            vn[i].idx = vi[i];
        }
        sort(vn.begin()+1,vn.begin()+55);
    }
    for(int i=1;i<=54;i++){
        cout<<vn[i].str;
        if(i<54) cout<<" ";
    }
    return 0;
}
