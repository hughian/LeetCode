#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<cstdio>
using namespace std;
int printf(const char*,...);
int scanf(const char*,...);
struct Student{
    int id;
    string name;
    int grade;
    Student():id(0),name(""),grade(0){}
    Student(int i,string n,int g):id(i),name(n),grade(g){}
};
vector<Student> vst;
int N,C;
bool cmp(Student &a,Student &b)
{
    if(C==1){
        return a.id < b.id;
    }else if(C==2){
        if(a.name == b.name)
            return a.id < b.id;
        return a.name < b.name;
    }else{
        if(a.grade == b.grade)
            return a.id < b.id;
        return a.grade < b.grade;
    }
}

int main()
{
    cin>>N>>C;
    int id,grade;
	char buf[10];
    string name;
    for(int i=0;i<N;i++){
		scanf("%d%s%d",&id,buf,&grade);
        name = string(buf);
        vst.push_back(Student(id,name,grade));
    }
    sort(vst.begin(),vst.end(),cmp);
    for(int i=0;i<(int)vst.size();i++)
        printf("%06d %s %d\n",vst[i].id,vst[i].name.c_str(),vst[i].grade);
    return 0;
}
