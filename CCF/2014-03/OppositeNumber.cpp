#include<iostream>

using namespace std;

class Solution{
public:
    void findOppositeNums(){
        int n;
        int cnt = 0;
        cin>>n;
        int a[n];
        for(int i=0;i<n;i++){
            cin>>a[i];
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(i==j) continue;
                if(a[i] + a[j] == 0 && a[i] != 0){
                    cnt++;
                    a[i] = 0;
                    a[j] = 0;
                }
            }
        }
        cout<<cnt;
    }
};

int main(){
    Solution s;
    s.findOppositeNums();
    return 0;
}
