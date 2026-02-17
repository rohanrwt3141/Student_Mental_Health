import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

def model_streamlit():

    st.title("Mental Stress Prediction ")
    # READING DATA
    df = pd.read_csv("/home/rohan/Desktop/student_mental_health_data_1000.csv")

    # PRINTING DATA
    # print(df.head())

    # LABEL ENCODER
    le_exam = LabelEncoder()
    le_assignment = LabelEncoder()
    le_academic = LabelEncoder()
    le_sleep = LabelEncoder()
    le_physical = LabelEncoder()
    le_screen = LabelEncoder()
    le_eating = LabelEncoder()
    le_family = LabelEncoder()
    le_living = LabelEncoder()
    le_friend = LabelEncoder()
    le_mental = LabelEncoder()

    # LABEL ENCODING
    df["exam_encoder"] = le_exam.fit_transform(df["Exam_Stress"])
    df["assignment_encoder"] = le_assignment.fit_transform(df["Assignment_Workload"])
    df["academic_encoder"] = le_academic.fit_transform(df["Academic_Performance"])
    df["sleep_encoder"] = le_sleep.fit_transform(df["Sleep_Duration"])
    df["physical_encoder"] = le_physical.fit_transform(df["Physical_Activity"])
    df["screen_encoder"] = le_screen.fit_transform(df["Screen_Time"])
    df["eating_encoder"] = le_eating.fit_transform(df["Eating_Habits"])
    df["family_encoder"] = le_family.fit_transform(df["Family_Support"])
    df["friends_encoder"] = le_friend.fit_transform(df["Friend_Support"])
    df["living_encoder"] = le_living.fit_transform(df["Living_Condition"])
    df["mental_encoder"] = le_mental.fit_transform(df["Mental_Health_Level"])

    # NEW DATAFRAME ARE CREATED
    X = df[["exam_encoder", "assignment_encoder", "academic_encoder", "sleep_encoder", "physical_encoder","screen_encoder", "eating_encoder", "family_encoder", "friends_encoder", "living_encoder"]]
    y = df["mental_encoder"]

    # print(df.head(), X.head())

    # TRAINING TESTING AND SPLITING THE DATA
    # print(df["exam_encoder"].head(10), df["Exam_Stress"].head(10))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)

    # MODEL IS CALLED
    model = RandomForestClassifier(random_state = 42)

    # MODEL IS TRAINED
    model.fit(X_train, y_train)

    # PREDICTION ON THE X_test DATA
    y_pred = model.predict(X_test)


    # PERFORMANCE MATRIES
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")
    # print(f"Classification Report : {classification_report(y_test, y_pred)}")

    # DATA FROM THE USER IS TAKEN
    exam = st.selectbox(f"Enter The Exam Stress :", list(le_exam.classes_))
    assignment = st.selectbox(f"Enter The Assignment Workload :", list(le_assignment.classes_))
    academic = st.selectbox(f"Enter The Academic Performance :", list(le_academic.classes_))
    sleep = st.selectbox(f"Enter The Sleep Duration :", list(le_sleep.classes_))
    physical = st.selectbox(f"Enter The Physical Activity Level :", list(le_physical.classes_))
    screen = st.selectbox(f"Enter The Screen Time :", list(le_screen.classes_))
    eating = st.selectbox(f"Enter Your Eating Habit :", list(le_eating.classes_))
    family = st.selectbox(f"Enter Your Family Support Level :", list(le_family.classes_))
    friend = st.selectbox(f"Enter Friend Support Level :", list(le_friend.classes_))
    living = st.selectbox(f"Enter The Living Condition :", list(le_living.classes_))

    # LABEL ENCODING IS PERFORMED IN THE INPUT DATA
    T_exam = le_exam.transform([exam])
    T_assignment = le_assignment.transform([assignment])
    T_academic = le_academic.transform([academic])
    T_sleep = le_sleep.transform([sleep])
    T_physical = le_physical.transform([physical])
    T_screen = le_screen.transform([screen])
    T_eating = le_eating.transform([eating])
    T_family = le_family.transform([family])
    T_friend = le_friend.transform([friend])
    T_living = le_living.transform([living])

    # NEW DATAFRAME IS CREATE FOR PREDICTION
    T_data = {"exam_encoder": T_exam, "assignment_encoder":T_assignment, "academic_encoder":T_academic, "sleep_encoder":T_sleep, "physical_encoder":T_physical, "screen_encoder":T_screen, "eating_encoder":T_eating, "family_encoder":T_family, "friends_encoder":T_friend, "living_encoder":T_living}
    T_df = pd.DataFrame(T_data)

    prediction = model.predict(T_df)

    # if(prediction == 2):
    #     return "Moderate", exam, assignment, academic, sleep, physical, screen, eating, family, friend , living
    # elif(prediction == 0):
    #     return "Heathy", exam, assignment, academic, sleep, physical, screen, eating, family, friend , living
    # elif(prediction == 1):
    #     return "High Stress", exam, assignment, academic, sleep, physical, screen, eating, family, friend , living
    # else:
    #     return "Error"

    if (prediction == 2):
        st.write("Moderate")
    elif (prediction == 0):
        st.write("Heathy")
    elif (prediction == 1):
        st.write("High Stress")
    else:
        st.write("Error")

    if(st.button("Solution")):

        # Sleep Cycle
        if (sleep == "More than 8 hours" or sleep == "7-8 hours"):
            st.write("Your Sleep Cycle is Very Good..... Keep It Up")
        else:
            st.write("7 to 8 hours sleep is required for a Student and your Sleeping Hours is ")
            st.write("I suggest you to watch this video and apply in your life ")
            st.video("https://www.youtube.com/watch?v=lIo9FcrljDk")

        # Exam Stress
        if (exam == "Low"):
            st.write("Your Exam Stress is Very Good..... Keep It Up")
        else:
            st.write("Low Exam Stress is required for a Student and your Exam Stress is ")
            st.write("I suggest you to watch this video and apply in your life ")
            st.video("https://www.youtube.com/watch?v=Ni3VEgbvuhU")


model_streamlit()