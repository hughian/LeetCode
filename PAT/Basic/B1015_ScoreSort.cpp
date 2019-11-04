#include<iostream>
#include<vector>
#include<cstdio>
#include<algorithm>
using namespace std;
class Student{
public:
    int id;
    int d,c,sum;
public:
    Student(int id_,int d_,int c_)
        :id(id_),d(d_),c(c_),sum(d_+c_){}
};
static bool MyCmp(const Student &s1,const Student & s2)
{
	return (s1.sum != s2.sum)?(s1.sum > s2.sum): \
				(s1.d != s2.d)?(s1.d > s2.d):(s1.id <= s2.id);
}
void print(vector<Student> &stn)
{
    for(unsigned i=0;i<stn.size();i++){
		printf("%d %d %d\n",stn[i].id,stn[i].d,stn[i].c);
		//使用Cout会超时，Cout效率低下。
        //cout<<stn[i].id<<" "<<stn[i].d<<" "<<stn[i].c<<endl;
    }
}

int main()
{
    int N,L,H;
    cin>>N>>L>>H;
    vector<Student> stn1,stn2,stn3,stn4;
    int id,d,c;
    for(int i=0;i<N;i++){
        cin>>id>>d>>c;
        if(d>=H && c>=H){
            stn1.push_back(Student(id,d,c));
        }
        else if(d>=H && c>=L)
            stn2.push_back(Student(id,d,c));
        else if(d>=c && c>=L)
            stn3.push_back(Student(id,d,c));
        else if(d>=L && c>=L)
            stn4.push_back(Student(id,d,c));
    }
    int ans = stn1.size() + stn2.size()+stn3.size() + stn4.size();
    cout<<ans<<endl;
    sort(stn1.begin(),stn1.end(),MyCmp);
    sort(stn2.begin(),stn2.end(),MyCmp);
    sort(stn3.begin(),stn3.end(),MyCmp);
    sort(stn4.begin(),stn4.end(),MyCmp);
    print(stn1);
    print(stn2);
    print(stn3);
    print(stn4);
    return 0;
}
