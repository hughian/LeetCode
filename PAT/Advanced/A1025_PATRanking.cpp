#include<iostream>
#include<vector>
#include<algorithm>
#include<string> 
using namespace std;
struct Testee{
    string id;
    int score;
    int location;
    int localr,rank;
    Testee():id(""),score(0),location(0),localr(0),rank(0){}
    Testee(string s,int n,int pos):id(s),score(n),location(pos),localr(0),rank(0){}
    bool operator < (const Testee & a)const{
        if(this->score == a.score)
            return this->id < a.id;
        return this->score > a.score;
    }
};
vector< vector<Testee> > vvi;
vector<Testee> vn;

int main()
{
    int n,k;
    cin>>n;
    string id;
    int score;
    vvi.resize(n);
    for(int i=0;i<n;i++){
        cin>>k;
        for(int j=0;j<k;j++){
            cin>>id>>score;
            vvi[i].push_back(Testee(id,score,i+1));
        }
        sort(vvi[i].begin(),vvi[i].end());
        vvi[i][0].localr = 1;
        vn.push_back(vvi[i][0]);
        for(int j=1;j<k;j++){
            if(vvi[i][j].score == vvi[i][j-1].score)
                vvi[i][j].localr = vvi[i][j-1].localr;
            else
                vvi[i][j].localr = j+1;
            vn.push_back(vvi[i][j]);
        }
    }
    sort(vn.begin(),vn.end());
    vn[0].rank = 1;
    for(int i=1;i<(int)vn.size();i++){
        if(vn[i].score == vn[i-1].score)
            vn[i].rank = vn[i-1].rank;
        else
            vn[i].rank = i+1;
    }
    cout<<vn.size()<<endl;
    for(int i=0;i<(int)vn.size();i++){
        cout<<vn[i].id<<" "<<vn[i].rank<<" "<<vn[i].location<<" "<<vn[i].localr<<endl;
    }
    return 0;
}
