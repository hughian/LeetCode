#include<iostream>
#define DEBUG
using namespace std;
int salaryRange[] = {3500,3500+1500,3500+4500,3500+9000,3500+35000,3500+55000,3500+80000};
int taxRate[] = {3,10,20,25,30,35,45};
const int SIZE = sizeof(salaryRange) / sizeof(int);
int range[SIZE];
class Solution{
public:
    int SalaryCalculate(){
        int t,s;
        range[0]= salaryRange[0];
        for(int i =1;i<SIZE;i++){
            range[i] = range[i-1] + (salaryRange[i] - salaryRange[i-1]) \
                       - (salaryRange[i] - salaryRange[i-1]) * taxRate[i-1] / 100;
        }
		#ifndef DEBUG //debug print to see the range
		for(int i = 0;i<SIZE;i++){
			cout<<range[i]<<" ";
		}
		cout<<endl;
		#endif
        cin>>t;
        int i;
        for(i=0;i<SIZE;i++){
            if(t <= range[i]) break;
        }

        if(i==0)
            s = t;
        else{
            s = salaryRange[i-1] + (t-range[i-1]) * 100/(100-taxRate[i-1]);
        }
        cout<<s<<endl;
        return 0;
    }
};

int main()
{
    Solution s;
    s.SalaryCalculate();
    return 0;
}
