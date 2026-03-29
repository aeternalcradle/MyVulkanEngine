class Singleton {
    private:
    Singleton(){
        cout<<"create singleton"<<endl;
    }
    ~Singleton(){
        cout<<"destroy singleton"<<endl;
    }
    public:
    static Singleton* getInstance(){
        static Singleton instance;
        return &instance;
    }
    void print(){
        cout<<"print singleton"<<endl;
    }

    Singleton(const Singleton&) = delete;
    Singleton& operator = (const Singleton&) = delete;

}
int main(){
    Singleton* singleton = Singleton::getInstance();
    singleton->print();
    return 0;
}