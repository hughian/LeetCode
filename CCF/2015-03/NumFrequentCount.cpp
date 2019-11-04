#include<iostream>
using namespace std;
int a[1001][2];

class Solution{
    void BubbleSort(){
        for(int i=1;i<1001;i++)
            for(int j=1000;j>i;j--){
				if(a[j][1] > a[j-1][1]){
					int tmp0 = a[j][0];
					int tmp1 = a[j][1];
					a[j][0] = a[j-1][0];
					a[j][1] = a[j-1][1];
					a[j-1][0] = tmp0;
					a[j-1][1] = tmp1;
				}
			}
    }

public:
    void NumFrequentCount(){
        for(int i=1;i<1001;i++){
                a[i][0] = i;
                a[i][1] = 0;
        }
        int n,tmp;
        cin>>n;
        for(int i=0;i<n;i++){
            cin>>tmp;
            a[tmp][1]++;
        }
        BubbleSort();
		for(int i=1;i<1001;i++){
            if(a[i][1])
                cout<<a[i][0]<<" "<<a[i][1]<<endl;
        }
    }
};
int main()
{
    Solution s;
    s.NumFrequentCount();
    return 0;
}
