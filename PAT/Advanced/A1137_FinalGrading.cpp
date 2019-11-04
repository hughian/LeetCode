#include<iostream>
#include<string>
#include<map>
#include<vector>
#include<algorithm>
using namespace std;
map<string,int> mp;
vector<string> vs;

int getindex(string sid){
    if(mp.count(sid)==0){
        mp[sid] = vs.size();
        vs.push_back(sid);
    }
    return mp[sid];
}
string getsid(unsigned int index){
    return index>=vs.size()?"":vs[index];
}

bool mycmp(vector<int> v1,vector<int> v2)
{
    if(v1[3]==v2[3])
        return getsid(v1[4]) < getsid(v2[4]); 
    else
        return v1[3]>v2[3];
}

int main()
{
    int pmn[3];
    cin>>pmn[0]>>pmn[1]>>pmn[2];
    string sid;
    int grade;
    vector< vector<int> > GL(30003,vector<int>(5,-1));
    
    for(int k = 0;k<3;k++){
        for(int i=0;i<pmn[k];i++){
            cin>>sid>>grade;
            int r = getindex(sid);
            GL[r][4] = r;
            GL[r][k] = grade;
        }
    }

    for(unsigned int i=0;i<mp.size();i++){
        if(GL[i][1]==-1)
            GL[i][3] = GL[i][2];
        else if(GL[i][1]>GL[i][2])
            GL[i][3] = 0.4 * GL[i][1] + 0.6 * GL[i][2] + 0.5;
        else
            GL[i][3] = GL[i][2];
    }
    /*
    for(unsigned i = 0;i<mp.size();i++){
        cout<<getsid(i)<<" "<<GL[i][0]<<" "<<GL[i][1]<<" ";
        cout<<GL[i][2]<<" "<<GL[i][3]<<" "<<GL[i][4]<<endl;
    }
    cout<<endl;
	*/
    sort(GL.begin(),GL.end(),mycmp);
    
    for(unsigned i = 0;i<mp.size();i++){
        if(GL[i][0]>=200 && GL[i][3]>=60){
            cout<<getsid(GL[i][4])<<" "<<GL[i][0]<<" "<<GL[i][1]<<" ";
            cout<<GL[i][2]<<" "<<GL[i][3]<<endl;
        }
    }


    return 0;
}
