#include<iostream>
#include<stdio.h>
using namespace std;

int mDays[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
class Solution{
    int IsLeapYear(int year){
        if(year%400 == 0 || (year%4 == 0 && year%100 != 0))
            return 1;
        return 0;
    }
    int getDay(int y){
        if(y<1850) return 0;
        int days = 0;
        for(int i=1850;i<y;i++){
            if(IsLeapYear(i))
                days += 366;
            else
                days += 365;
        }
        return (days+1)%7 + 1;
    }
    //y -- year
    //m -- month
    //d -- first day of year(y)
    int getMonthDay(int y,int m,int d){
        int days = 0;
        for(int i=0;i<m-1;i++){
            if(i==1)
                days += mDays[1]+IsLeapYear(y);
            else
                days += mDays[i];
        }
        return (days + d-1)%7 + 1;
    }
public:
    void HolidayDate(){
        int a,b,c,y1,y2;
        cin>>a>>b>>c>>y1>>y2;
        for(int y=y1; y <= y2 ; y++){
            int day = getDay(y); //first day of year y(01/01) 
            int md = getMonthDay(y,a,day); //y 年 a 月的第一天是周几
            
            int monthDays ;
            if(a == 2) 
                monthDays = mDays[1] + IsLeapYear(y);
            else 
                monthDays = mDays[a-1];
            
            int ans = ((b-1)*7 + (c+7-md)%7) + 1;
            if(ans > monthDays)
                cout<<"none"<<endl;
            else
                printf("%d/%02d/%02d\n",y,a,ans);
        }
    }
};

int main()
{
    Solution s;
    s.HolidayDate();
    return 0;
}
