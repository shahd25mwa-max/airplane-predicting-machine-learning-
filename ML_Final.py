import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score , mean_absolute_error , accuracy_score, classification_report
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle as pkl
import streamlit as st



data = pd.read_csv(r"C:\Users\Asus\OneDrive\airplanepro_MLfinal\delay\Flight_delay.csv")
data2=data.copy()
#####################################################dealing with missing values####################################################

data = data.drop(["Origin" , "Dest", "Cancelled"], axis=1)
data.dropna(subset=['Org_Airport'], inplace=True)
data.dropna(subset=['Dest_Airport'], inplace=True)
# print(data.isna().sum())
# print(data["Org_Airport"].value_counts())
# print(data["Org_Airport"].isna().sum())

#####################################################Encoding objects###################################################
org_air = data['Org_Airport'].unique()
print(sorted(list(org_air)))
org_air = sorted(list(org_air))
des_air = data['Dest_Airport'].unique()
print(sorted(list(des_air)))
des_air = sorted(list(des_air))
data.info()


###########################1-date########################

data["Date"]=pd.to_datetime(data["Date"], format='mixed')
data["year"] = data["Date"].dt.year
data["month"] = data["Date"].dt.month
data["day"] = data["Date"].dt.day
data.drop(["Date"], axis=1, inplace=True)
year=sorted(list(data["year"].unique()))
month=sorted(list(data["month"].unique()))
day=sorted(list(data["day"].unique()))

############################2- UniqueCarrier###########################

carri_enc= LabelEncoder()
data["UniqueCarrier"] = carri_enc.fit_transform(data["UniqueCarrier"])
# data.info()

############################3-Airline###########################
airline=sorted(list(data['Airline'].unique()))
air_enc = LabelEncoder()
data["Airline"] = air_enc.fit_transform(data["Airline"])


############################4-TailNum airplane plate num###########################

tail_enc= LabelEncoder()
data["TailNum"] = tail_enc.fit_transform(data["TailNum"])

############################5- Origin -> original out airport###########################

org_enc = LabelEncoder()
data["Org_Airport"] = org_enc.fit_transform(data["Org_Airport"])

############################ 6- dest --> the entered airport###########################

dest_enc = LabelEncoder()
data["Dest_Airport"] = dest_enc.fit_transform(data["Dest_Airport"])

############################7- CancellationCode###########################
# ما الها داعي بس المهم الفكره
data["CancellationCode"] = data["CancellationCode"].map({"N":0})
# data.info()


#************************************************************** ------ First Model ------ ****************************************************************
########################### ----- Arrival Time Prediction ----- ###########################

x = data[["year", "day" , "DepTime" , "FlightNum" , "Airline" , "Org_Airport" , "Dest_Airport"]]
y = data["ArrDelay"]
x_train , x_test , y_train , y_test = train_test_split(x, y , test_size = 0.2, random_state =42)
#********------------ Algorithms -----------*******
#+++++++++++++  XGboost ++++++++++++
# m_xg = xgb.XGBRegressor(device='cuda')
# m_xg.fit(x_train, y_train)
# y_pred = m_xg.predict(x_test)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("%%%%%MAE: " , mae)   # %%%%%MAE:  1.9026968479156494
# print("R@4444444444:",r2)   #  R@4444444444: 0.9853411316871643  ####### results aren't good enough

m_LR = LinearRegression()
m_LR.fit(x_train, y_train)
y_pred = m_LR.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("%%%%%MAE: " , mae) #  %%%%%MAE:  4.770002040631723e-05

print("R@4444444444:",r2)  #  R@4444444444: 0.9999999945878245

#####linearregression has the best results

#************ --- Saving files  -------
with open("arrival_model.pkl", "wb") as f:
    pkl.dump(m_LR, f)
with open("original_airport.pkl", "wb") as f:
    pkl.dump(org_enc, f)
with open("destination_airport.pkl", "wb") as f:
    pkl.dump(dest_enc, f)
with open("Airline.pkl", "wb") as f:
    pkl.dump(air_enc, f)
#**********************************************- Second Model -*********************************
####################3-------------- Price prediction ------------#$#####################

df = pd.read_csv(r"C:\Users\Asus\OneDrive\airplanepro_MLfinal\fuel\airline_ticket_prices.csv")
df2= df.copy()
# print(df["yoy_price_change_pct"].unique())
df2['yoy_price_change_pct'] = df2['yoy_price_change_pct'].fillna(df2['yoy_price_change_pct'].mean())
df['yoy_price_change_pct'] = df['yoy_price_change_pct'].fillna(df['yoy_price_change_pct'].mean())
#no missing values
# month encoding

df["month"]=pd.to_datetime(df["month"], format='mixed')
df["year"] = df["month"].dt.year
df["month"] = df["month"].dt.month

#conflict_phase  ---> how realted is it with wars or pandamics that happens in the world

cofl_enc = LabelEncoder()
df["conflict_phase"] = cofl_enc.fit_transform(df["conflict_phase"])


# airline encoding

airline2_enc = LabelEncoder()
df["airline"] = airline2_enc.fit_transform(df["airline"])
#iata_code
iata_enc = LabelEncoder()
df["iata_code"] = iata_enc.fit_transform(df["iata_code"])
# df.info()
#
# print("con" , df["country"].unique())
# print("reg" , df["region"].unique())
# print("airline type ", df["airline_type"].unique())
# print(" rout " , df["route_class"].unique())

# country
con_enc = LabelEncoder()
df["country"] = con_enc.fit_transform(df["country"])
#reg
reg_enc = LabelEncoder()
df["region"] = reg_enc.fit_transform(df["region"])
#airline type
airT_enc = LabelEncoder()
df["airline_type"] = airT_enc.fit_transform(df["airline_type"])

# route_class

rou_enc= LabelEncoder()
df["route_class"]= rou_enc.fit_transform(df["route_class"])
df.info()
# done encoding
print(df.head())
# data spliting
xp = df.drop(["total_fare_usd" ,  "base_fare_usd" ] , axis=1)
yp = df["total_fare_usd"]
xp_train, xp_test, yp_train, yp_test = train_test_split(xp, yp , test_size = 0.2, random_state = 42)

# p_lr= LinearRegression()
# p_lr.fit(xp_train, yp_train)
# yp_pred = p_lr.predict(xp_test)
# acc = r2_score(yp_test,yp_pred )
# kn= KNeighborsRegressor()
# kn.fit(xp_train, yp_train)
# yp_pred_0 = kn.predict(xp_test)
# acc0= r2_score(yp_test,yp_pred )
# print("KNN",acc0)
# print("LR r2 score " , acc)
# cross = cross_val_score(p_lr , xp_test , yp_test , cv=5, scoring="r2")
p_rF= RandomForestRegressor()
p_rF.fit(xp_train, yp_train)
yp_pred_1= p_rF.predict(xp_test)
acc1= r2_score(yp_test,yp_pred_1 )
print("RandomForest",acc1)
# DT_P  = DecisionTreeRegressor()
# DT_P.fit(xp_train, yp_train)
# yp_pred = DT_P.predict(xp_test)
# acc = r2_score(yp_test,yp_pred )
# print("Decision tree score " , acc)
###########KNN 0.9831765379773693   LR r2 score  0.9831765379773693    RandomForest 0.990036364132371######### the results shows that the random forest is the best model


## saving the model
with open("ticket_prices.pkl", "wb") as f:
    pkl.dump(p_rF, f)



# $$$$$$$$$$ ------- Streamlit program ----------$$$$$$$$
st.title("Flight Analytics & Prediction System")
st.sidebar.header("choose what do you want to predict?")
sides = st.sidebar.radio("here : ", ["expected price .", "expected arrival time."])

# 1- predicting price
if sides == "expected price .":
    st.header("Expected Ticket Price")
    st.write("please enter the flight  details:")

    sel_mon_p= st.selectbox("month of the flight ", sorted(list(df["month"].unique())))
    sel_air_p=st.selectbox("select the Airline : ", sorted(list( df2["airline"].unique() )))
    sel_con_p= st.selectbox("select the Country :" , sorted(list(df2["country"].unique())))
    sel_phase = st.selectbox("select Conflict Phase :", sorted(list(df2["conflict_phase"].unique())))
    sel_route = st.selectbox("select Route Class : " , sorted(list( df2["route_class"].unique() )))

    km = st.number_input("Average Route KM", value=1500)

    brent = st.number_input("Brent Crude USD", value=75.0)

    jet_fuel =st.number_input("Jet Fuel USD Barrel", value=95.0)

    tax = st.number_input("Taxes and Fees USD", value=50.0)
    surcharge = st.number_input("Fuel Surcharge USD", value=30.0)

    enc_air_p = airline2_enc.transform([sel_air_p])[0]
    enc_con_p = con_enc.transform([sel_con_p])[0]
    enc_phase = cofl_enc.transform([sel_phase])[0]
    enc_route = rou_enc.transform([sel_route])[0]

    info_p = [[sel_mon_p,enc_phase,enc_air_p,df["iata_code"].median(), enc_con_p,df["region"].median(),
               df["airline_type"].median(),enc_route,km,surcharge,tax,brent,jet_fuel,df["load_factor_pct"].median(),
               df["fuel_cost_pct_opex"].median(),df["yoy_price_change_pct"].median(),df["year"].median()]]

    if st.button("Predict Price"):
        pre_price = p_rF.predict(info_p)
        st.write("the expected ticket price is : $", int(pre_price[0]),"USD")


elif sides == "expected arrival time.":
    st.header("Expected Arrival Time")
    st.write("please enter your flight detiels ")
    sel_year = st.selectbox("what's the date of your flight ?   \n year", year)
    sel_mon = st.selectbox("month", month)
    sel_day = st.selectbox("day", day)

    col1, col2 = st.columns(2)
    with col1:
        sel_orline = st.selectbox("select the airport you are getting out of :", org_air)
        sel_deline = st.selectbox("select the airport you are heading to  :", des_air)
    with col2:
        sel_air = st.selectbox("select the Airline :", airline)

        f_num = st.number_input("Flight Number", value=101)
        dep_time = st.number_input("Departure Time (HHMM)", value=1200)

    enc_org = org_enc.transform([sel_orline])[0]
    enc_dest = dest_enc.transform([sel_deline])[0]
    enc_air = air_enc.transform([sel_air])[0]

    info = [[sel_year, sel_day, dep_time, f_num, enc_air, enc_org, enc_dest]]

    if st.button("Predict Delay"):
        pre = m_LR.predict(info)
        st.write("the result is this :", int(pre[0]),"Min")