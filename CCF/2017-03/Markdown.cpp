#include<iostream>
#include<string>
#include<vector>
using namespace std;

class Solution{
    string line;
    vector<string> ans;
    bool Bflg;
private:
    void dealwith(string str){
        int len = str.length();
        for(int i=0;i<len;i++){
            if(str[i] == '_'){
                int r = str.find('_',i+1);
                string tmp = "<em>" + str.substr(i+1,r-i-1) + "</em>";
                dealwith(tmp);
                i = r;
            }
            else if(str[i] == '['){
                int r = str.find(']',i+1);
                string text = str.substr(i+1,r-i-1);

                int l = str.find('(',r+1);
                int rlink = str.find(')',r+1);
                string link = str.substr(l+1,rlink-l-1);
                string tmp = "<a href=\""+link + "\">" + text + "</a>";
                dealwith(tmp);
                i = rlink;
            }
            else {
                cout<<str[i];
            }
        }
    }
    void Head(){
        int i=0;
        int lv;
        string str = ans.at(0);
        while(str[i]=='#') i++;
        lv = i;
        while(str[i]==' ') i++;
        string tmp = str.substr(i,str.size()-i);
        cout<<"<h"<<lv<<">";
        dealwith(tmp);
        cout<<"</h"<<lv<<">"<<endl;
        ans.erase(ans.begin());
    }
    void List(){
        cout<<"<ul>"<<endl;
        string str;
        vector<string>::iterator it = ans.begin();
        while(it!=ans.end()){
            str = *it;
            int i = 1;
            while(str[i] == ' ') i++;
            str = str.substr(i,str.size()-i);
            cout<<"<li>";
            dealwith(str);
            cout<<"</li>"<<endl;
            ans.erase(it);
        }
        cout<<"</ul>"<<endl;
    }
    void Text(){
        string str;
        cout<<"<p>";
        vector<string>::iterator it = ans.begin();
        while(it != ans.end()){
            str = *it;
            dealwith(str);
            if(it+1 == ans.end())
                cout<<"</p>";
            cout<<endl;
            ans.erase(it);
        }
    }
public:
    void Markdown(){
        //freopen("test.txt","r",stdin);
        while(getline(cin,line)){
            if(line == ""){
                if(ans.size()==0)
                    continue;
                switch(ans[0][0]){
                    case 0:
                        continue;break;
                    case '#':
                        Head();break;
                    case '*':
                        List();break;
                    default:
                        Text();
                }
                continue;
            }
            ans.push_back(line);
        }
        if(!ans.empty()){
            if(ans[0][0] == '#')
                Head();
            else if(ans[0][0] == '*')
                List();
            else
                Text();
        }
    }
};

int main()
{
    Solution s;
    s.Markdown();
    return 0;
}
