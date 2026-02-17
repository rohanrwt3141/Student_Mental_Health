import pandas as pd
from suggestPassword_backend import suggestPassword

def sign_in():
    print("# --------------- Sign In --------------- #")

    email = input("Enter your username")

    suggestion = int(input("Enter 1 if you want Password Suggestion : "))

    if(suggestion == 1):
        password = suggestPassword()
        print(password)
    else:
        password = input("Enter your password")


    data = {"email": [email], "password": [password]}
    df = pd.DataFrame(data, columns=["email", "password"])
    df.to_csv("/home/rohan/Desktop/Information.csv", mode = "a+",header = False, index = False, columns = ["email", "password"])

    print("Successfully Sign in")

    if(email == "" or email[-10:] != "@gmail.com" or password == ""):
        print("Please enter a valid email address")

sign_in()


