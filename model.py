import numpy as np
import pandas as pd
from tkinter import *
from PIL import ImageTk
ds = pd.read_csv(r"D:\Project\Student Performance Prediction\student-mat.csv", sep=";")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC # Support Vector Machine Classifier model

def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.2, random_state=17)

def confuse(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # print("\nConfusion Matrix: \n", cm)
    fpr(cm)
    ffr(cm)

""" False Pass Rate """
def fpr(confusion_matrix):
    fp = confusion_matrix[0][1]
    tf = confusion_matrix[0][0]
    rate = float(fp) / (fp + tf)
    print("False Pass Rate: ", rate)

""" False Fail Rate """
def ffr(confusion_matrix):
    ff = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]
    rate = float(ff) / (ff + tp)
    print("False Fail Rate: ", rate)

    return rate

model=""

""" Train Model and Print Score """
def train_and_score(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)

    clf = Pipeline([
        ('reduce_dim', SelectKBest(chi2, k=2)),
        ('train', LinearSVC(C=100))
    ])

    scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=2)
    print("Mean Model Accuracy:", np.array(scores).mean())

    clf.fit(X_train, y_train)
    print(X_test)
    confuse(y_test, clf.predict(X_test))
    print()
    return clf

""" Main Program """
def main():
    global model
    print("\nStudent Performance Prediction")

    df=ds.drop(columns=['famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','activities','nursery','higher','romantic','famrel','Dalc','Walc'])
    
    class_le = LabelEncoder()
    for column in df[["school", "sex", "address","paid","internet"]].columns:
        df[column] = class_le.fit_transform(df[column].values)

    # Encode G1, G2, G3 as pass or fail binary values
    for i, row in df.iterrows():
        if row["G1"] >= 10:
            df["G1"][i] = 1
        else:
            df["G1"][i] = 0

        if row["G2"] >= 10:
            df["G2"][i] = 1
        else:
            df["G2"][i] = 0

        if row["G3"] >= 10:
            df["G3"][i] = 1
        else:
            df["G3"][i] = 0

    # Target values are G3
    y = df.pop("G3")

    # Feature set is remaining features
    X = df

    print("\n\nModel Accuracy Knowing G1 & G2 Scores")
    print("=====================================")
    model=train_and_score(X, y)
    

   # Remove grade report 2
    X.drop(["G2"], axis = 1, inplace=True)
    print("\n\nModel Accuracy Knowing Only G1 Score")
    print("=====================================")
    train_and_score(X, y)

    # Remove grade report 1
    X.drop(["G1"], axis=1, inplace=True)
    print("\n\nModel Accuracy Without Knowing Scores")
    print("=====================================")
    train_and_score(X, y)
    
    root=Tk()
    root.title("Student Performance Prediction")
    root.geometry("1350x700+0+0")
    bg_i=ImageTk.PhotoImage(file=r"D:\Project\Student Performance Prediction\Tracking_Headline.png")
    bg_l = Label(root, image=bg_i,bg="dimgray")
    bg_l.place(x=10, y=200,width=500,height=300)
    bg_i2=ImageTk.PhotoImage(file=r"D:\Project\Student Performance Prediction\suggestions.png")
    bg_l1 = Label(root, image=bg_i2,bg="dimgray")
    bg_l1.place(x=1000, y=200,width=240,height=230)
    T=Text(root,height=1,width=35,font=("bold",24),bg="dimgray",bd=0,fg="mistyrose2")
    T.pack()
    T.insert(END,'Welcome to Student Performance Prediction')
    root.configure(background="dimgray")
    l=Label(root,text='Enter the details:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=550,y=40,width=200,height=25)
    l=Label(root,text='School:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=70,width=200,height=25)
    e1=Entry(root)
    e1.place(x=750,y=70,width=100,height=25)
    l=Label(root,text='Sex:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=100,width=200,height=25)
    svar=StringVar(root)
    s=["M","F"]
    w=OptionMenu(root,svar,*s)
    w.place(x=750,y=100,width=100,height=25)
    l=Label(root,text='Age:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=130,width=200,height=25)
    e2=Entry(root)
    e2.place(x=750,y=130,width=100,height=25)
    add=StringVar(root)
    s2=["U","R"]
    l=Label(root,text='Address:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=160,width=200,height=25)
    w=OptionMenu(root,add,*s2)
    w.place(x=750,y=160,width=100,height=25)
    medu=StringVar(root)
    fedu=StringVar(root)
    s1=['1','2','3','4']
    s3=['1','2','3','4','5','6','7','8','9','10']
    s4=['Yes','No']
    s5=['1','2','3','4','5']
    s6=list(range(0,21))
    l=Label(root,text='Mother Education:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=190,width=200,height=25)
    w=OptionMenu(root,medu,*s1)
    w.place(x=750,y=190,width=100,height=25)
    
    l=Label(root,text='Father Education:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=220,width=200,height=25)
    w=OptionMenu(root,fedu,*s1)
    w.place(x=750,y=220,width=100,height=25)
    
    tt=StringVar(root)
    l=Label(root,text='Travel Time:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=250,width=200,height=25)
    w=OptionMenu(root,tt,*s3)
    w.place(x=750,y=250,width=100,height=25)
    
    st=StringVar(root)
    l=Label(root,text='Study Time:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=280,width=200,height=25)
    w=OptionMenu(root,st,*s3)
    w.place(x=750,y=280,width=100,height=25)
    
    f=StringVar(root)
    l=Label(root,text='Failures:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=310,width=200,height=25)
    w=OptionMenu(root,f,*s3)
    w.place(x=750,y=310,width=100,height=25)
    
    pc=StringVar(root)
    l=Label(root,text='Paid Courses:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=340,width=200,height=25)
    w=OptionMenu(root,pc,*s4)
    w.place(x=750,y=340,width=100,height=25)
    
    inte=StringVar(root)
    l=Label(root,text='Internet:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=370,width=200,height=25)
    w=OptionMenu(root,inte,*s4)
    w.place(x=750,y=370,width=100,height=25)
    
    ft=StringVar(root)
    l=Label(root,text='Free Time:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=400,width=200,height=25)
    w=OptionMenu(root,ft,*s3)
    w.place(x=750,y=400,width=100,height=25)
    
    ot=StringVar(root)
    l=Label(root,text='Out Time:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=430,width=200,height=25)
    w=OptionMenu(root,ot,*s3)
    w.place(x=750,y=430,width=100,height=25)
    
    h=StringVar(root)
    l=Label(root,text='Health:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=460,width=200,height=25)
    w=OptionMenu(root,h,*s5)
    w.place(x=750,y=460,width=100,height=25)
    
    ab=StringVar(root)
    l=Label(root,text='Absences:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=490,width=200,height=25)
    w=OptionMenu(root,ab,*s3)
    w.place(x=750,y=490,width=100,height=25)
    
    sc1=StringVar(root)
    l=Label(root,text='Mid1 Score:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=520,width=200,height=25)
    w=OptionMenu(root,sc1,*s6)
    w.place(x=750,y=520,width=100,height=25)
    
    sc2=StringVar(root)
    l=Label(root,text='Mid2 Score:-',font=('bold',16),fg="white",bg="dimgray")
    l.place(x=554,y=550,width=200,height=25)
    w=OptionMenu(root,sc2,*s6)
    w.place(x=750,y=550,width=100,height=25)
    def success():
        tp=Toplevel()
        tp.title("Success")
        tp.geometry("400x400+0+0")
        tp.configure(background="dimgray")
        bg_i=ImageTk.PhotoImage(file=r"D:\Project\Student Performance Prediction\success.png")
        bg_l = Label(tp, image=bg_i,bg="dimgray")
        bg_l.place(x=40, y=10,width=300,height=300)
        l=Label(tp,text='Keep it Up......',font=('bold',16),fg="white",bg="dimgray")
        l.place(x=10,y=270,width=400,height=25)
        l=Label(tp,text='You will Succeed',font=('bold',16),fg="white",bg="dimgray")
        l.place(x=10,y=300,width=400,height=25)
        tp.mainloop()
    def failure():
        tp=Toplevel()
        tp.title("Failure")
        tp.geometry("400x400+0+0")
        tp.configure(background="dimgray")
        bg_i=ImageTk.PhotoImage(file=r"D:\Project\Student Performance Prediction\failure.png")
        bg_l = Label(tp, image=bg_i,bg="dimgray")
        bg_l.place(x=40, y=10,width=300,height=300)
        l=Label(tp,text='Work Hard.....',font=('bold',16),fg="white",bg="dimgray")
        l.place(x=10,y=270,width=420,height=25)
        l=Label(tp,text='You will Fail',font=('bold',16),fg="white",bg="dimgray")
        l.place(x=10,y=300,width=400,height=25)
        tp.mainloop()
    
    def suggest():
        di={}
        di['school']=[e1.get()]
        di['sex']=[svar.get()]
        di['age']=[int(e2.get())]
        di['address']=[add.get()]
        di['Medu']=[int(medu.get())]
        di['Fedu']=[int(fedu.get())]
        di['traveltime']=[int(tt.get())]
        di['studytime']=[int(st.get())]
        di['failures']=[int(f.get())]
        di['paid']=[pc.get()]
        di['internet']=[inte.get()]
        di['freetime']=[int(ft.get())]
        di['goout']=[int(ot.get())]
        di['health']=[int(h.get())]
        di['absences']=[int(ab.get())]
        di['G1']=[int(sc1.get())]
        di['G2']=[int(sc2.get())]
        df1=pd.DataFrame(di)
        class_le = LabelEncoder()
        for column in df1[["school", "sex", "address","paid","internet"]].columns:
            df1[column] = class_le.fit_transform(df1[column].values)
        for i, row in df1.iterrows():
            if row["G1"] >= 10:
                df1["G1"][i] = 1
            else:
                df1["G1"][i] = 0
    
            if row["G2"] >= 10:
                df1["G2"][i] = 1
            else:
                df1["G2"][i] = 0
        pred=model.predict(df1)
        print(df1)
        print(pred[0])
        if pred[0]==0:
            failure()
        else:
            success()
        
    btn4=Button(root,command=suggest,text="Predict",compound=LEFT,font=("Industry Inc Detail Fill", 20, "bold"), bg="white", fg="dimgray")
    btn4.place(x=650,y=600,height=50,width=120)
    root.mainloop()
main()

