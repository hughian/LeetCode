#include<iostream>
using namespace std;

int mDays[12] = {31,28,31,30,31,30,31,31,30,31,30,31}; 

class Solution{
public:
    void DateCount(){
        int y,d;
        int i =0;
        cin>>y>>d;
        while(d>0){
            if(i==2)
                d -= mDays[i]+isLeapYear(y);
            else
                d -= mDays[i];
            i++;
        }
        cout<<i<<endl;
        if(i-1 == 2)
            cout<<d+mDays[i-1]+isLeapYear(y);
        else
            cout<<d+mDays[i-1];
    }
private:
    int isLeapYear(int year){
        if((year % 4 == 0 && year % 100 != 0) || year % 400 ==0)
            return 1;
        return 0;
    }
};

int main()
{
    Solution s;
    s.DateCount();
    return 0;
}
