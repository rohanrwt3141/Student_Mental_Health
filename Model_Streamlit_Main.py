import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

def model_streamlit():

    st.markdown("""<h1 style = "color: #FA5C5C;"><strong>Mental Stress Prediction</strong></h1>""", unsafe_allow_html = True)

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
    st.markdown(""" <h5 style = "color: #FD8A6B; "> <strong> Enter The Exam Stress : </strong> </h5> """, unsafe_allow_html = True)
    exam = st.selectbox("", list(le_exam.classes_))

    st.markdown(""" <h5 style = "color: #FD8A6B; "> <strong> Enter The Assignment Workload : </strong> </h5> """, unsafe_allow_html=True)
    assignment = st.selectbox("", list(le_assignment.classes_))

    st.markdown(""" <h5 style = "color: #FD8A6B; "> <strong> Enter The Academic Performance : </strong> </h5> """, unsafe_allow_html=True)
    academic = st.selectbox("", list(le_academic.classes_))

    st.markdown(""" <h5 style = "color: #FD8A6B; "> <strong> Enter The Sleep Duration : </strong> </h5> """, unsafe_allow_html=True)
    sleep = st.selectbox("", list(le_sleep.classes_))

    st.markdown(""" <h5 style = "color: #FD8A6B; "> <strong> Enter The Physical Activity Level : </strong> </h5> """, unsafe_allow_html=True)
    physical = st.selectbox("", list(le_physical.classes_))

    st.markdown(""" <h5 style = "color: #FD8A6B; "> <strong> Enter The Screen Time : </strong> </h5> """, unsafe_allow_html=True)
    screen = st.selectbox(f"", list(le_screen.classes_))

    st.markdown("""<h5 style = "color: #FD8A6B; "> <strong> Enter Your Eating Habit : </strong> </h5> """, unsafe_allow_html=True)
    eating = st.selectbox(f"", list(le_eating.classes_))

    st.markdown("""<h5 style = "color: #FD8A6B; "> <strong> Enter Your Family Support Level : </strong> </h5> """, unsafe_allow_html=True)
    family = st.selectbox(f"", list(le_family.classes_), key= "family")

    st.markdown("""<h5 style = "color: #FD8A6B; "> <strong> Enter Friend Support Level : </strong> </h5> """, unsafe_allow_html=True)
    friend = st.selectbox(f"", list(le_friend.classes_))

    st.markdown("""<h5 style = "color: #FD8A6B; "> <strong> Enter The Living Condition : </strong> </h5> """, unsafe_allow_html=True)
    living = st.selectbox(f"", list(le_living.classes_))

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
    if(st.button("Predict")):
        if (prediction == 2):
            st.markdown(""" <h5 style = "color: yellow; "> <strong> <em> Moderate </em> </strong> </h5> """, unsafe_allow_html=True)

        elif (prediction == 0):
            st.markdown(""" <h5 style = "color: green; "> <strong> <em> Heathy </em> </strong> </h5> """, unsafe_allow_html=True)

        elif (prediction == 1):
            st.markdown(""" <h5 style = "color: red; "> <strong> <em> High Stress </em> </strong> </h5> """, unsafe_allow_html=True)
        else:
            st.write("Error")

    if(st.button("Solution")):

        # Sleep Cycle
        st.markdown(""" <h4 style = "color: black; background: #ccc; "> About Your Sleep Cycle  </h4> """, unsafe_allow_html=True)

        if (sleep == "More than 8 hours" or sleep == "7-8 hours"):
            st.markdown(""" <h5 style = "color: green; background: #ccc;"> Your Sleep Cycle is Very Good..... Keep It Up </h5>""", unsafe_allow_html = True)
        else:
            st.markdown("""<h4 style = "color: blue; background: #ccc;" > 7 to 8 hours sleep is required for a Student. </h4>""", unsafe_allow_html = True)
            st.markdown("""<h5 style = "color: black; background: #ccc;" >I suggest you to watch this video and apply in your life </h5>""", unsafe_allow_html = True)
            st.video("https://www.youtube.com/watch?v=lIo9FcrljDk")
            st.markdown("""<a href = "https://www.youtube.com/results?search_query=Improve+Sleep+Cycle+" target = "_blank" ><button style=\'padding:10px 20px; font-size:16px;'> More Videos </button></a>""", unsafe_allow_html=True)

        # CREATE A LINE
        st.markdown("""<h3 style = "color: white;">------------------------------------------------------------------------------</h3>""", unsafe_allow_html=True)

        # Exam Stress
        st.markdown(""" <h4 style = "color: black; background: #ccc; "> About Your Exam Stress  </h4> """, unsafe_allow_html=True)

        if (exam == "Low"):
            st.markdown(""" <h5 style = "color: green; background: #ccc"> Your Exam Stress is Very Good..... Keep It Up </h5>""", unsafe_allow_html=True)
        elif(exam == "Very Low"):
            st.markdown(""" <p style = "color: black; background: #ccc; "> Your Exam Stress is Very Low  </p> """, unsafe_allow_html=True)
            st.markdown(""" <p style = "color: black; background: #ccc; "> Experiencing very low exam stress 😊 may feel comfortable and relaxed, but it can sometimes create hidden challenges. When stress is extremely low, a student may become overconfident 😌 and assume that preparation is already sufficient. This overconfidence can lead to procrastination ⏳, where study tasks are delayed because there is no sense of urgency. Gradually, the seriousness toward the exam may decrease 📉, and important topics might not receive proper attention. As the exam date approaches, this relaxed attitude can suddenly turn into last-minute pressure 😰, forcing rushed revision and unnecessary stress. </p>""", unsafe_allow_html=True)

        else:
            st.markdown(""" <h5 style = "color: blue; background: #ccc;" > Low Exam Stress is required for a Student </h5> """, unsafe_allow_html = True)
            st.markdown(""" <h5 style = "color: black; background: #ccc;" > I suggest you to watch this video and apply in your life </h5> """, unsafe_allow_html = True)
            st.video("https://www.youtube.com/watch?v=Ni3VEgbvuhU")
            st.markdown(""" <a href = "https://www.youtube.com/results?search_query=Serious+video+for+Exam+if+we+have+very+low+stress" target = "_blank" ><button style="padding:10px 20px; font-size:16px;"> More Videos </button> </a> """, unsafe_allow_html=True)

        # CREATE A LINE
        st.markdown("""<h3 style = "color: white;">------------------------------------------------------------------------------</h3>""", unsafe_allow_html=True)

        # Assignment Workload
        st.markdown(""" <pre> <h4 style = "color: black; background: #ccc; ">                About Your Assignment Workload  </h4> </pre>""", unsafe_allow_html=True)
        if(assignment == "Moderate"):
            st.markdown(""" <p style = "color: black; background: #ccc; "> Your Assignment Workload is Moderate.  </p> """, unsafe_allow_html=True)
        elif(assignment == "Very Light" or assignment == "Light"):
            pass
        else:
            st.markdown(
                """ <h5 style = "color: blue; background: #ccc;" > Low Exam Stress is required for a Student </h5> """, unsafe_allow_html=True)
            st.markdown(
                """ <h5 style = "color: black; background: #ccc;" > I suggest you to watch this video and apply in your life </h5> """, unsafe_allow_html=True)
            st.video("https://www.youtube.com/watch?v=31LZeMC5l3o")
            st.markdown(
                """ <a href = "https://www.youtube.com/results?search_query=how+to+balance+assignment+workload+of+college" target = "_blank" ><button style="padding:10px 20px; font-size:16px;"> More Videos </button> </a> """, unsafe_allow_html=True)

            # Physical Activity
            st.markdown(
                """ <pre> <h4 style = "color: black; background: #ccc; ">                About Your Physical Activity  </h4> </pre>""",
                unsafe_allow_html=True)
            if (physical == "Daily" or physical == "Regularly"):
                st.markdown(""" <p style = "color: black; background: #ccc; "> Your Physical Activity is Good for Healthy Stress.  </p> """, unsafe_allow_html=True)

            else:
                st.markdown(
                    """ <h5 style = "color: blue; background: #ccc;" > Physical Activity is required for the Healthy Life </h5> """,
                    unsafe_allow_html=True)
                st.markdown(
                    """ <h5 style = "color: black; background: #ccc;" > I suggest you to watch this video and apply in your life </h5> """,
                    unsafe_allow_html=True)
                st.video("https://www.youtube.com/watch?v=UzWd8ynGLEM")
                st.markdown(
                    """ <a href = "https://www.youtube.com/results?search_query=importance+of+physical+activity" target = "_blank" ><button style="padding:10px 20px; font-size:16px;"> More Videos </button> </a> """,
                    unsafe_allow_html=True)

                # Screen Time
                st.markdown(
                    """ <pre> <h4 style = "color: black; background: #ccc; ">                About Your Screen Time  </h4> </pre>""", unsafe_allow_html=True)
                if (screen  == "2-4 hours" or screen == "Less than 2 hours"):
                    st.markdown(""" <p style = "color: black; background: #ccc; "> Your Screen Time is Good for the Productive Use .  </p> """, unsafe_allow_html=True)

                else:
                    st.markdown(
                        """ <h5 style = "color: blue; background: #ccc;" > Screen Time of less than 4 hours only for productive Use is required for a Student </h5> """, unsafe_allow_html=True)
                    st.markdown(
                        """ <h5 style = "color: black; background: #ccc;" > I suggest you to watch this video and apply in your life </h5> """, unsafe_allow_html=True)
                    st.video("https://www.youtube.com/watch?v=rx1l51qxGbQ")
                    st.markdown(
                        """ <a href = "https://www.youtube.com/results?search_query=Screen+Time+Problem" target = "_blank" ><button style="padding:10px 20px; font-size:16px;"> More Videos </button> </a> """, unsafe_allow_html=True)

                    # Eating Habit
                    st.markdown(
                        """ <pre> <h4 style = "color: black; background: #ccc; ">                About Your Eating Habit  </h4> </pre>""",
                        unsafe_allow_html=True)
                    if (eating == "Very Healthy" or eating == "Mostly Healthy"):
                        st.markdown(
                            """ <p style = "color: black; background: #ccc; "> Your Eating Habit is Good ... Keep It Up .  </p> """,
                            unsafe_allow_html=True)

                    else:
                        st.markdown(
                            """ <h5 style = "color: blue; background: #ccc;" > Healthy Eating Habit is required for a Student </h5> """,
                            unsafe_allow_html=True)
                        st.markdown(
                            """ <h5 style = "color: black; background: #ccc;" > I suggest you to watch this video and apply in your life </h5> """,
                            unsafe_allow_html=True)
                        st.video("https://www.youtube.com/watch?v=OEM4_7hVQfU")
                        st.markdown(
                            """ <a href = "https://www.youtube.com/results?search_query=Eating+Habit" target = "_blank" ><button style="padding:10px 20px; font-size:16px;"> More Videos </button> </a> """,
                            unsafe_allow_html=True)

                        # Family Support
                        st.markdown(
                            """ <pre> <h4 style = "color: black; background: #ccc; ">                About Your Eating Habit  </h4> </pre>""",
                            unsafe_allow_html=True)
                        if (family == "High" or family == "Very High"):
                            st.markdown(
                                """ <p style = "color: black; background: #ccc; "> Your Family Support is Good ... Keep It Up .  </p> """,
                                unsafe_allow_html=True)

                        else:
                            st.markdown(
                                """ <h5 style = "color: blue; background: #ccc;" > High Family Support is required for a Student </h5> """,
                                unsafe_allow_html=True)
                            st.markdown(
                                """ <h5 style = "color: black; background: #ccc;" > If family support is limited, try:

            👥 Close friends – Share your thoughts honestly.

            🎓 Mentors/Teachers – They can guide academically and emotionally.

            💬 Counselors – Professional support helps in stress control.

            🌐 Positive online communities – Study groups or growth-focused groups. </h5> """,
                                unsafe_allow_html=True)

                            # Friend Support
                            st.markdown(
                                """ <pre> <h4 style = "color: black; background: #ccc; ">                About Your Eating Habit  </h4> </pre>""",
                                unsafe_allow_html=True)
                            if (friend == "Very Supportive" or friend == "Supportive"):
                                st.markdown(
                                    """ <p style = "color: black; background: #ccc; "> Your Friend Support is Good ... Keep It Up .  </p> """,
                                    unsafe_allow_html=True)

                            else:
                                st.markdown(
                                    """ <h5 style = "color: blue; background: #ccc;" > High Friend Support is required for a Student </h5> """,
                                    unsafe_allow_html=True)
                                st.markdown(
                                    """ <h5 style = "color: black; background: #ccc;" > When friends are less supportive:

            📓 Start journaling your thoughts

            🎯 Set small daily goals

            🧘 Practice meditation or deep breathing

            📚 Focus on skill improvement

            👉 Self-growth builds confidence that doesn’t depend on others. </h5> """,
                                    unsafe_allow_html=True)


model_streamlit()