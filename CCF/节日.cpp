#include<bits/stdc++.h>
using namespace std;
inline bool isLeapYear(int year){
	return year%400==0||(year%4==0&&year%100);
}
int days(int month,bool leap){
	switch(month){
		case 1:return 0;
		case 2:return 31;
		case 3:return leap?60:59;
		case 4:return leap?91:90;
		case 5:return leap?121:120;
		case 6:return leap?152:151;
		case 7:return leap?182:181;
		case 8:return leap?213:212;
		case 9:return leap?244:243;
		case 10:return leap?274:273;
		case 11:return leap?305:304;
		case 12:return leap?335:334;
	}
}
int monthDay(int month,bool leap){
	switch(month){
		case 1:case 3:case 5:case 7:case 8:case 10:case 12:return 31;
		case 4:case 6:case 9:case 11:return 30;
		case 2:return leap?29:28;
	}
}
int main()
{
	int month,week,day,y1,y2;
	scanf("%d%d%d%d%d",&month,&week,&day,&y1,&y2);
	//if(y1>y2) swap(y1,y2);//no pro bro
	int leapCount=0;
	for(int i=1850;i<y1;i++)
		if(isLeapYear(i)) leapCount++;
	for(int i=y1;i<=y2;i++){
		bool isLeap=isLeapYear(i);
		int totDays=(i-1850)*365+leapCount+days(month,isLeap);
		int weekDay=(2+totDays)%7;weekDay==0?weekDay=7:;
		int ans=(day-weekDay<0)?(day-weekDay+week*7+1):(day-weekDay+(week-1)*7+1);
		if(ans<=monthDay(month,isLeap)) printf("%04d/%02d/%02d\n",i,month,ans);
		else printf("none\n");
		if(isLeap) leapCount++;
	}
	return 0;
}
