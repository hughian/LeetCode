#include<iostream>
#include<vector>
#include<string>
using namespace std;

int main()
{
    string Mm_name,Mm_ID,Fm_name,Fm_ID;
    bool ff=true,mf=true;
    int N,Mm_grade,Fm_grade,grade;
    cin>>N;
    string name,id,gender;
    for(int i=0;i<N;i++){
        cin>>name>>gender>>id>>grade;
        if(gender=="F"){
            if(ff){
                Fm_name = name;
                Fm_ID = id;
                Fm_grade = grade;
                ff = false;
            }else{
                if(grade > Fm_grade){
                    Fm_name = name;
                    Fm_ID = id;
                    Fm_grade = grade;
                }
            }
        }else if(gender == "M"){
            if(mf){
                Mm_name = name;
                Mm_ID = id;
                Mm_grade = grade;
                mf = false;
            }else{
                if(grade < Mm_grade){
                    Mm_name = name;
                    Mm_ID = id;
                    Mm_grade = grade;
                }
            }
        
        }
    }

    if(ff)
        cout<<"Absent"<<endl;
    else
        cout<<Fm_name+" "+Fm_ID<<endl;
    if(mf)
        cout<<"Absent"<<endl;
    else
        cout<<Mm_name+" "+Mm_ID<<endl;
    if(ff || mf)
        cout<<"NA"<<endl;
    else
        cout<<Fm_grade-Mm_grade<<endl;
    return 0;
}
