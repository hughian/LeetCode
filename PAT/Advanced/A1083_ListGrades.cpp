#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
using namespace std;
struct Student{
    string name,id;
    int grade;
    Student(string n,string s,int g):name(n),id(s),grade(g){}
    bool operator < (const Student& stu){
        return this->grade > stu.grade;
    }
};

int main()
{
    vector<Student> vs;
    int n;
    cin>>n;
    string name,id;
    int g;
    for(int i=0;i<n;i++){
        cin>>name>>id>>g;
        vs.push_back(Student(name,id,g));
    }
    int g1,g2;
    cin>>g1>>g2;
    sort(vs.begin(),vs.end());
    bool flag = false;
    for(int i=0;i<(int)vs.size();i++){
        if(vs[i].grade >=g1 && vs[i].grade<= g2){
            flag = true;
            cout<<(vs[i].name+" "+vs[i].id)<<endl;
        }
    }
    if(!flag)
        cout<<"NONE";
    return 0;
}

