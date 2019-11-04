#include<iostream>
#include<string>
using namespace std;
typedef struct Date{
    int year,month,day;
    Date(int y,int m,int d)
        :year(y),month(m),day(d){}
    Date():year(0),month(0),day(0){}
    Date &operator = (const Date &d){
        this->year = d.year;
        this->month = d.month;
        this->day = d.day;
    }   
}DATE;

bool cmp(DATE d1,DATE d2)
{
	if(d1.year == d2.year){
		if(d1.month == d2.month)
			return d1.day > d2.day;
		return d1.month>d2.month;
	}
	return d1.year > d2.year;
}

int main()
{
    int n;
    cin>>n;
    string name,birth;
    DATE date;
    string maxName,minName;
    DATE MIN(1814,9,6);
    DATE MAX(2014,9,6);
    DATE maxdate = MIN,mindate = MAX;
    int cnt=0;
    for(int i=0;i<n;i++){
        cin>>name>>birth;
        date.year = 0;
        for(int k=0;k<4;k++)
            date.year = date.year * 10 + (birth[k]-'0');
        date.month = (birth[5] - '0') * 10 + (birth[6] - '0');
        date.day = (birth[8]-'0') * 10 + (birth[9] - '0');
        if(cmp(date,MAX)|| cmp(MIN,date))
            continue;
        cnt++;

        if(cmp(date,maxdate)){
            maxdate = date;
            maxName = name;
        }
        if(cmp(mindate,date)){
            mindate = date;
            minName = name;
        }
    }
	
	//注意这里cnt可能为零个的情况
	if(cnt)
		cout<<cnt<<" "<<minName<<" "<<maxName;
    else
		cout<<"0";
	return 0;
}
