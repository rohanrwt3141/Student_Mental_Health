import pandas as pd

def log_in():

    data = pd.read_csv("/home/rohan/Desktop/Information.csv")

    df = pd.DataFrame(data)
    # print(df.columns)

    email = input("Enter your Email Address:")

    password = input("Enter your Password:")

    flag = False
    for i in range(len(data)):
        if(df["email"].iloc[i] == email and df["password"].iloc[i] == password):
            print("Successfully logged in")
            flag  = True
    if(flag == False):
        print("Either your email address is not valid, try again")

log_in()