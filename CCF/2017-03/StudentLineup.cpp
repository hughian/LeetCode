#include<iostream>
#include<vector>
using namespace std;

class Solution{
    vector<int> a;
    vector< vector<int> > cmd;
public:
    void studentLineup(){
        int n,m;
        cin>>n>>m;
        a.resize(n);
        for(int i=0;i<n;i++)
            a.at(i) = i+1;
        cmd.resize(m);
        for(int i=0;i<m;i++){
            cmd.at(i).resize(2);
            cin>>cmd[i][0]>>cmd[i][1];
        }
        for(int i=0;i<m;i++){
            int k;
            for(k =0;k<n;k++){
                if(a[k] == cmd[i][0]){
                    break;
                }
            }
            if(cmd[i][1] == 0)
                ;
            else if(cmd[i][1] > 0){ //forward
                int tmp = a[k];
                int t;
                for(t = k;t<k+cmd[i][1];t++){
                    a[t] = a[t+1];
                }
                a[t] = tmp;
            }else{
                int tmp = a[k];
                int t;
                for(t=k;t>k+cmd[i][1];t--){
                    a[t] = a[t-1];
                }
                a[t]=tmp;
            }
        }
        for(int i=0;i<n;i++){
            cout<<a[i]<<" ";
        }
    }
};

int main()
{
    Solution s;
    s.studentLineup();
    return 0;
}

